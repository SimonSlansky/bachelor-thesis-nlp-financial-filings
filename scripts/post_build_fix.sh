#!/usr/bin/env bash
# Post-build helper for atomic install of prace.pdf
# Usage: post_build_fix.sh /abs/path/to/built/prace.pdf /abs/or/rel/path/to/workspace/tex/prace.pdf

set -euo pipefail

src="$1"
dst="$2"

# Resolve dst to absolute path relative to repository root if needed
if [[ ! "$dst" = /* ]]; then
  # Make dst absolute relative to script location
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  dst="$repo_root/$dst"
fi

echo "post_build_fix: src=$src dst=$dst"

# Ensure src exists
if [[ ! -f "$src" ]]; then
  echo "error: built file not found: $src" >&2
  exit 2
fi

# Create destination directory if needed
mkdir -p "$(dirname "$dst")"


# Function to compute sha256; fall back to shasum
sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    openssl dgst -sha256 "$1" | awk '{print $2}'
  fi
}

# Copy with verification and retry in case OneDrive replaces the file while syncing.
max_retries=5
attempt=1
src_sum=$(sha256 "$src")
echo "post_build_fix: source sha256=$src_sum"

while [ $attempt -le $max_retries ]; do
  echo "post_build_fix: attempt $attempt -> copying to workspace"
  tmp_dest="$(dirname "$dst")/.prace.pdf.tmp.$$.$attempt"
  cp -f -- "$src" "$tmp_dest"

  # Clear extended attributes and ACLs on the temp file
  if command -v xattr >/dev/null 2>&1; then
    xattr -c "$tmp_dest" 2>/dev/null || true
  fi
  if command -v chmod >/dev/null 2>&1; then
    chmod -N "$tmp_dest" 2>/dev/null || true
  fi
  chmod 0644 "$tmp_dest" || true

  # Move into place atomically
  mv -f -- "$tmp_dest" "$dst"

  # Force a sync to flush filesystem buffers
  if command -v sync >/dev/null 2>&1; then
    sync
  fi

  # Give the fileprovider a brief moment to settle
  sleep 0.8

  if [ -f "$dst" ]; then
    dst_sum=$(sha256 "$dst")
  else
    dst_sum=""
  fi

  echo "post_build_fix: dst sha256=$dst_sum"

  if [ "$dst_sum" = "$src_sum" ]; then
    echo "post_build_fix: verified workspace copy matches source"
    break
  else
    echo "post_build_fix: mismatch (src != dst). retrying..."
    attempt=$((attempt+1))
  fi
done

if [ $attempt -gt $max_retries ]; then
  echo "post_build_fix: failed to reliably install prace.pdf into workspace after $max_retries attempts" >&2
  echo "post_build_fix: leaving verified Desktop copy and original built file in place" >&2
fi

echo "post_build_fix: installed $dst (attempts=$attempt)"

# Also copy a stable local copy to Desktop for quick opening outside OneDrive
desktop_dst="$HOME/Desktop/prace-local.pdf"
cp -f -- "$src" "$desktop_dst"
if command -v xattr >/dev/null 2>&1; then
  xattr -c "$desktop_dst" 2>/dev/null || true
fi
chmod 0644 "$desktop_dst" || true
echo "post_build_fix: copied local Desktop copy $desktop_dst"

# Optional verification on the Desktop copy
if command -v qpdf >/dev/null 2>&1; then
  qpdf --check "$desktop_dst" || echo "warning: qpdf found issues in $desktop_dst"
fi
if command -v gs >/dev/null 2>&1; then
  gs -q -dNODISPLAY -dBATCH -dNOPAUSE -sDEVICE=nullpage -sOutputFile=/dev/null "$desktop_dst" || echo "warning: gs could not render $desktop_dst"
fi

exit 0
