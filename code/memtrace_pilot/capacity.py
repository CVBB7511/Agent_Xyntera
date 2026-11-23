"""Use registry layer metadata to estimate disk before a large Docker Hub pull."""

import concurrent.futures
import json
import struct
import subprocess
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def layer_sizes(ref):
    name, sha = ref.split("@")
    url = "https://auth.docker.io/token?" + urlencode(
        {"service": "registry.docker.io", "scope": "repository:" + name + ":pull"}
    )
    token = json.load(urlopen(url, timeout=25))["token"]
    base = "https://registry-1.docker.io/v2/" + name + "/"

    def get(path, extra=None):
        headers = {"Authorization": "Bearer " + token}
        headers.update(extra or {})
        return urlopen(Request(base + path, headers=headers), timeout=35)

    accept = "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"
    with get("manifests/" + sha, {"Accept": accept}) as f:
        manifest = json.load(f)
    with get("blobs/" + manifest["config"]["digest"]) as f:
        ids = json.load(f)["rootfs"]["diff_ids"]

    def measure(layer):
        if "gzip" not in layer["mediaType"]:
            raise ValueError("unknown compression format")
        end = layer["size"] - 1
        with get("blobs/" + layer["digest"], {"Range": f"bytes={end - 3}-{end}"}) as f:
            if (
                f.status != 206
                or f.getheader("Content-Range", "").split("/")[0]
                != f"bytes {end - 3}-{end}"
            ):
                raise ValueError("registry did not honor four-byte gzip footer request")
            return struct.unpack("<I", f.read(4))[0]

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        sizes = list(pool.map(measure, manifest["layers"]))
    return dict(zip(ids, sizes, strict=True))


def installed_layers():
    ids = subprocess.check_output(
        ["docker", "image", "ls", "-q", "--no-trunc"], text=True
    ).split()
    if not ids:
        return set()
    inspect = json.loads(
        subprocess.check_output(["docker", "image", "inspect", *set(ids)], text=True)
    )
    return {k for img in inspect for k in img["RootFS"]["Layers"]}
