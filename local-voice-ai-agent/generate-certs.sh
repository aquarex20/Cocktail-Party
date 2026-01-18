#!/bin/bash
# Generate self-signed SSL certificates for HTTPS
# Required for browser microphone access on non-localhost deployments

set -e

CERT_DIR="certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Check if certificates already exist
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "Certificates already exist in ./$CERT_DIR/"
    echo "To regenerate, delete the existing certificates first:"
    echo "  rm -rf $CERT_DIR"
    exit 0
fi

# Create certs directory
mkdir -p "$CERT_DIR"

# Generate self-signed certificate
echo "Generating self-signed SSL certificates..."
openssl req -x509 -newkey rsa:4096 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 365 \
    -nodes \
    -subj "/CN=localhost"

echo ""
echo "Certificates generated successfully in ./$CERT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Rebuild and start the containers:"
echo "     docker compose build"
echo "     docker compose up -d"
echo ""
echo "  2. Access the app via HTTPS:"
echo "     https://your-server-ip:7860"
echo ""
echo "  3. Accept the browser's certificate warning (expected for self-signed certs)"
echo ""
echo "Note: For production with a domain, replace these with Let's Encrypt certificates."
