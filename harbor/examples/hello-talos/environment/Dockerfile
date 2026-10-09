FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git config --system user.name "harbor" \
    && git config --system user.email "harbor@example.invalid" \
    && git config --system --add safe.directory /app \
    && git init -q /app \
    && printf '# hello-talos\n' > /app/README.md \
    && git -C /app add README.md \
    && git -C /app commit -q -m "baseline"
