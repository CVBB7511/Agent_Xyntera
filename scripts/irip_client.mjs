// Local controller transport; credentials stay in a closure and stdin pipe.
import { spawn } from 'node:child_process';

export function createIripClient(password, workerPath) {
  let credential = password;
  return {
    async call(requests) {
      if (!credential) throw new Error('Client credential cleared');
      const child = spawn('wsl.exe', [
        '-d', 'Ubuntu-20.04', '--exec', 'python3', '-u', workerPath,
      ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
      let stdout = '', stderr = '';
      child.stdout.on('data', data => { stdout += data.toString(); });
      child.stderr.on('data', data => { stderr += data.toString(); });
      const result = new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', code => resolve({
          code,
          messages: stdout.trim().split('\n').filter(Boolean).map(JSON.parse),
          stderr: stderr.includes('Traceback') ? stderr : '[WSL startup warning omitted]',
        }));
      });
      child.stdin.end([
        JSON.stringify({ password: credential }),
        ...requests.map((request, i) => JSON.stringify({ id: i + 1, ...request })),
        JSON.stringify({ id: requests.length + 1, action: 'close' }), '',
      ].join('\n'));
      return result;
    },
    clear() { credential = null; },
  };
}
