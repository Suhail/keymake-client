---
name: here-now
description: Publish files and folders to the web instantly via here.now. Use when asked to "publish this", "host this", "deploy this", "share this on the web", "make a website", or "put this online". Outputs a live URL at {slug}.here.now.
---

# here.now - Instant Web Hosting

Publish any file or folder to get a live URL. No account required (anonymous publishes expire in 24h).

## Workflow

1. Create your site files (HTML, CSS, JS, images, etc.)
2. Publish using the 3-step API below
3. Return the live URL to the user

## Publishing API

### Step 1: Create the publish

Declare all files you want to publish. Get the byte size of each file first.

```bash
# Get file sizes
INDEX_SIZE=$(wc -c < index.html)
STYLE_SIZE=$(wc -c < style.css)

# Create the publish
curl -s -X POST https://here.now/api/v1/publish \
  -H "Content-Type: application/json" \
  -d "{
    \"files\": [
      {\"path\": \"index.html\", \"size\": $INDEX_SIZE, \"contentType\": \"text/html; charset=utf-8\"},
      {\"path\": \"style.css\", \"size\": $STYLE_SIZE, \"contentType\": \"text/css; charset=utf-8\"}
    ]
  }" > /tmp/publish_response.json

cat /tmp/publish_response.json
```

The response contains:
- `siteUrl` - the live URL (e.g. `https://abc-xyz.here.now/`)
- `slug` - the site identifier
- `upload.versionId` - needed for finalization
- `upload.uploads[]` - array of presigned URLs, one per file (in same order as `files` array)

### Step 2: Upload files to presigned URLs

Extract the presigned URLs from the response and PUT each file:

```bash
# Parse response (using python for reliable JSON parsing)
SITE_URL=$(python3 -c "import json; d=json.load(open('/tmp/publish_response.json')); print(d['siteUrl'])")
SLUG=$(python3 -c "import json; d=json.load(open('/tmp/publish_response.json')); print(d['slug'])")
VERSION_ID=$(python3 -c "import json; d=json.load(open('/tmp/publish_response.json')); print(d['upload']['versionId'])")
UPLOAD_URL_0=$(python3 -c "import json; d=json.load(open('/tmp/publish_response.json')); print(d['upload']['uploads'][0]['url'])")
UPLOAD_URL_1=$(python3 -c "import json; d=json.load(open('/tmp/publish_response.json')); print(d['upload']['uploads'][1]['url'])")

# Upload each file
curl -s -X PUT "$UPLOAD_URL_0" -H "Content-Type: text/html; charset=utf-8" --data-binary @index.html
curl -s -X PUT "$UPLOAD_URL_1" -H "Content-Type: text/css; charset=utf-8" --data-binary @style.css
```

### Step 3: Finalize

```bash
curl -s -X POST "https://here.now/api/v1/publish/$SLUG/finalize" \
  -H "Content-Type: application/json" \
  -d "{\"versionId\": \"$VERSION_ID\"}"
```

The site is now live at `$SITE_URL`.

## Content Type Reference

| Extension | Content-Type |
|-----------|-------------|
| `.html`   | `text/html; charset=utf-8` |
| `.css`    | `text/css; charset=utf-8` |
| `.js`     | `application/javascript; charset=utf-8` |
| `.json`   | `application/json` |
| `.svg`    | `image/svg+xml` |
| `.png`    | `image/png` |
| `.jpg`    | `image/jpeg` |
| `.gif`    | `image/gif` |
| `.ico`    | `image/x-icon` |
| `.txt`    | `text/plain; charset=utf-8` |
| `.webp`   | `image/webp` |
| `.woff2`  | `font/woff2` |

## Tips

- Always include an `index.html` as the entry point
- Create all files in a working directory first, then publish them all at once
- The presigned URLs in `upload.uploads[]` are in the same order as the `files` array in your request
- Build attractive, well-designed sites with modern CSS (flexbox/grid, good typography, responsive)
- After publishing, always share the live URL with the user
