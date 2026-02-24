#!/usr/bin/env bash
# Blackhole Setup Script for AI Party
# Clones Blackhole, builds multiple targets (BlackHole2ch, BlackHole2chPrime, etc.),
# and installs .driver bundles to /Library/Audio/Plug-Ins/HAL
#
# Usage: ./blackhole_setup.sh [num_agents]
#   num_agents: 2, 3, or 4 (default: 2)
#
# Requires: Xcode (xcodebuild), git. Run from local-voice-ai-agent directory.
# macOS only.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BLACKHOLE_DIR="$REPO_DIR/third_party/BlackHole"
HAL_PLUGINS="/Library/Audio/Plug-Ins/HAL"

NUM_AGENTS="${1:-2}"

# Target configs: (driver_display_name, bundle_id) - order matters for 2-agent setup
get_targets() {
  case "$1" in
    2) echo "BlackHole2ch:audio.existential.BlackHole2ch BlackHole2chPrime:audio.existential.BlackHole2chPrime" ;;
    3) echo "BlackHole2ch:audio.existential.BlackHole2ch BlackHole2chPrime:audio.existential.BlackHole2chPrime BlackHole2ch3:audio.existential.BlackHole2ch3" ;;
    4) echo "BlackHole2ch:audio.existential.BlackHole2ch BlackHole2chPrime:audio.existential.BlackHole2chPrime BlackHole2ch3:audio.existential.BlackHole2ch3 BlackHole2ch4:audio.existential.BlackHole2ch4" ;;
    *) echo "Usage: $0 [2|3|4]"; exit 1 ;;
  esac
}

echo "=== Blackhole Setup for $NUM_AGENTS agents ==="

# Clone if needed
if [ ! -d "$BLACKHOLE_DIR" ]; then
  echo "Cloning Blackhole..."
  mkdir -p "$(dirname "$BLACKHOLE_DIR")"
  git clone --depth 1 https://github.com/ExistentialAudio/BlackHole.git "$BLACKHOLE_DIR"
else
  echo "Blackhole repo exists, pulling latest..."
  (cd "$BLACKHOLE_DIR" && git pull --depth 1 2>/dev/null || true)
fi

# Build each target (Blackhole outputs BlackHole.driver, we rename per target)
for spec in $(get_targets "$NUM_AGENTS"); do
  DRIVER_NAME="${spec%%:*}"
  BUNDLE_ID="${spec#*:}"
  echo ""
  echo "Building $DRIVER_NAME..."

  # Build with custom driver name and bundle ID (matches Blackhole create_installer.sh pattern)
  xcodebuild \
    -project "$BLACKHOLE_DIR/BlackHole.xcodeproj" \
    -target BlackHole \
    -configuration Release \
    CONFIGURATION_BUILD_DIR="$BLACKHOLE_DIR/build" \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    GCC_PREPROCESSOR_DEFINITIONS='$GCC_PREPROCESSOR_DEFINITIONS kNumber_Of_Channels=2 kPlugIn_BundleID=\"'"$BUNDLE_ID"'\" kDriver_Name=\"'"$DRIVER_NAME"'\"' \
    build 2>&1 | tail -10

  # Output is build/BlackHole.driver - rename to our target name
  if [ ! -d "$BLACKHOLE_DIR/build/BlackHole.driver" ]; then
    echo "ERROR: Build failed - BlackHole.driver not found"
    exit 1
  fi

  DRIVER_BUNDLE="$BLACKHOLE_DIR/build/${DRIVER_NAME}.driver"
  rm -rf "$DRIVER_BUNDLE"
  mv "$BLACKHOLE_DIR/build/BlackHole.driver" "$DRIVER_BUNDLE"
  echo "  Built: $DRIVER_BUNDLE"
done

# Install to HAL (requires sudo)
echo ""
echo "Installing to $HAL_PLUGINS (requires sudo)..."
for spec in $(get_targets "$NUM_AGENTS"); do
  DRIVER_NAME="${spec%%:*}"
  DRIVER_PATH="$BLACKHOLE_DIR/build/${DRIVER_NAME}.driver"
  if [ -d "$DRIVER_PATH" ]; then
    sudo cp -R "$DRIVER_PATH" "$HAL_PLUGINS/"
    echo "  Installed: ${DRIVER_NAME}.driver"
  fi
done

echo ""
echo "Restarting CoreAudio..."
sudo killall -9 coreaudiod 2>/dev/null || true
sleep 2

echo ""
echo "=== Done! ==="
echo "Next: Create multi-output devices in Audio MIDI Setup (see AI Party tab instructions)."
