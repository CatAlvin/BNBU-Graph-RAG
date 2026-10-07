"""Fetch release assets with SHA-256 verification and bounded archive extraction."""
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def main():
    manifest = json.loads((ROOT / 'assets-manifest.json').read_text())
    for item in manifest['assets']:
        target = (ROOT / item['destination']).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError('Asset destination escapes the project.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != item['sha256']:
            raise SystemExit('Existing asset has a different checksum: ' + target.name)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            if not item['url'].startswith('https://github.com/CatAlvin/'):
                raise ValueError('Unexpected release URL.')
            request = urllib.request.Request(item['url'], headers={'User-Agent': 'project-asset-downloader'})
            received = 0
            try:
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > item['bytes']:
                            raise ValueError('Asset exceeds its recorded size.')
                        output.write(chunk)
                if received != item['bytes'] or digest(temporary) != item['sha256']:
                    raise ValueError('Asset size or SHA-256 mismatch.')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        if item.get('zip'):
            with zipfile.ZipFile(target) as archive:
                if sum(member.file_size for member in archive.infolist()) > 256 * 1024 * 1024:
                    raise ValueError('Archive expands beyond the sample data limit.')
                for member in archive.infolist():
                    destination = (ROOT / member.filename).resolve()
                    if not destination.is_relative_to(ROOT) or destination.relative_to(ROOT).parts[0] not in {'features','data'}:
                        raise ValueError('Unexpected archive path.')
                    if member.is_dir():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open('wb') as output:
                        shutil.copyfileobj(source, output)
        print('Verified:', item['name'])

if __name__ == '__main__':
    main()
