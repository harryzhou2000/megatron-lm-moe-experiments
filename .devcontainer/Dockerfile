FROM nvcr.io/nvidia/pytorch:26.03-py3

# Install basic dev tools
# RUN apt-get update && apt-get install -y \
#     git \
#     vim \
#     sudo \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y clangd-19

RUN update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100

# Optional: create non-root user
# RUN useradd -ms /bin/bash vscode \
#     && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# USER vscode