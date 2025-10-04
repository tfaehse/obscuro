# Configuration Files

This directory contains configuration files for different environments:

- `default.toml` - Default configuration with balanced settings
- `dev.toml` - Development configuration optimized for speed
- `production.toml` - Production configuration optimized for quality

## Usage

### CLI
```bash
# Use development config
blur-cli --config config/dev.toml image input.jpg

# Use production config
blur-cli --config config/production.toml video input.mp4
```

### API
```bash
# Start API with a specific config file
blur-api --config config/production.toml
```

## Custom Configuration

Copy one of these files and modify it for your needs:

```bash
cp config/default.toml config/my-config.toml
# Edit config/my-config.toml
blur-cli --config config/my-config.toml image input.jpg
```
