## Nginx + lua (v1, current version)

we now use nginx + lua by openresty, providing a high-performance, production ready routing layer for multiple LLM backends using OpenResty (Nginx + Lua). 

### Overview

```bash
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Client    │─────▶│  OpenResty       │─────▶│  Backend 1      │
│  (API Call) │      │  (Router)        │      │  (Qwen@8000)    │
└─────────────┘      │                  │      └─────────────────┘
                     │  - Model Mapping │
                     │  - Load Balancing│      ┌─────────────────┐
                     │  - Health Checks │─────▶│  Backend 2      │
                     │  - Error Handling│      │  (Llama@8001)   │
                     └──────────────────┘      └─────────────────┘
```

## Installation

1. Install OpenResty

```bash
# Add repository
wget -O - https://openresty.org/package/pubkey.gpg | sudo apt-key add -
echo "deb http://openresty.org/package/ubuntu $(lsb_release -sc) main" | \
    sudo tee /etc/apt/sources.list.d/openresty.list

# Install
sudo apt-get update
sudo apt-get install openresty
```

1. Deployment

```bash
# Create directory
sudo mkdir -p /usr/local/openresty/nginx/conf/sites-available
sudo mkdir -p /usr/local/openresty/nginx/conf/sites-enabled

# Copy Config file
sudo cp <your config file> /usr/local/openresty/nginx/conf/sites-available/vllm

# Enable the site
sudo ln -s /usr/local/openresty/nginx/conf/sites-available/vllm \
           /usr/local/openresty/nginx/conf/sites-enabled/vllm
```

1. Modify `/usr/local/openresty/nginx/conf/nginx.conf` 

```bash
http {
    # ... Others ...
    
    # Lua settings
    lua_package_path "/usr/local/openresty/lualib/?.lua;;";
    lua_shared_dict model_cache 10m;
    
    # Include Site Configuration
    include /usr/local/openresty/nginx/conf/sites-enabled/*;
}
```

1. Enable services

```bash
# test openresty config
sudo openresty -t

# Start
sudo systemctl start openresty

# Enable auto-start
sudo systemctl enable openresty

# reload openresty
sudo openresty -s reload
```

### API Examples

```bash
# check service status
curl http://freeinference.org/health

# list all models
curl http://freeinference.org/v1/models | jq

# Chat with Qwen3-Coder
curl -X POST http://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Chat with llama4-scout
curl -X POST http://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/meta-llama_Llama-4-Scout-17B-16E", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

## Nginx (v0, abandoned)

```bash
sudo vim /etc/nginx/sites-available/vllm
sudo nginx -t
sudo systemctl reload nginx

# to test the endpoint
curl http://freeinference.org/v1/models
```