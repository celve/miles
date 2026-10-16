# syntax=docker/dockerfile:1.7
#
# The desktop image for every harbor_cua task: Xvfb, a window manager, Debian's real
# Chromium package, and the tools computer-1's runtime shells out to. Modelled on
# Harbor's own examples/tasks/computer-1-click-calibration/environment/Dockerfile.
#
# Debian rather than Ubuntu because its `chromium` apt package is a real binary rather
# than a snap stub, and no KVM, qcow2 or systemd is involved -- which is what lets this
# run as an ordinary container on a sandbox provider.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        curl \
        dbus-x11 \
        fonts-liberation \
        imagemagick \
        procps \
        scrot \
        webp \
        wmctrl \
        x11-utils \
        xauth \
        xdotool \
        xfce4-session \
        xfwm4 \
        xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp /var/tmp \
    && mkdir -p /tmp/.ICE-unix && chmod 1777 /tmp/.ICE-unix \
    && mkdir -p /logs && chmod 777 /logs

WORKDIR /app

# The page is inlined as base64 rather than copied in: the sandbox deployment these
# tasks run on has no build-context upload endpoint, so a copy instruction cannot
# resolve there (AgentENV#23).
RUN echo '__PAGE_BASE64__' | base64 -d > /app/page.html
