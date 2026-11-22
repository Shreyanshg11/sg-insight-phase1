# SG-Insight frontend — Vite/React SPA.
#
# This Dockerfile is for a generic Docker/Compose self-host. On Render
# specifically, the frontend deploys as a native static site instead (see
# render.yaml + README's Deployment section), so this image is not used there.
# It is still useful for local docker-compose parity and any other host that
# wants one container per service.

# ---- build stage ----
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
# Vite inlines VITE_* vars at BUILD time, not runtime — must be supplied here,
# not just at `docker run`. client.js does `${VITE_API_URL}/api/dataset/...`,
# so this must be an ORIGIN (empty string = same-origin), never a path like
# "/api" — that would double up to "/api/api/dataset/...". Empty default
# matches the nginx reverse-proxy config below for local docker-compose use.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ---- serve stage ----
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
