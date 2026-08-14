// Deliberately minimal. This exists to satisfy PWA installability criteria
// and give some offline resilience for the static shell — it is NOT trying
// to be a full offline-first app, because this app's whole point is live
// analysis of data the user just uploaded, which can't be meaningfully
// cached anyway.
//
// Critical rule: /api/* is NEVER intercepted. Every request to the backend
// goes straight to the network, always. Caching an API response here would
// mean showing stale profiling/cleaning/analysis results after the user's
// data or session state has actually changed — worse than no offline
// support at all.
const CACHE_NAME = "sg-insight-shell-v1";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;   // don't touch cross-origin
  if (url.pathname.startsWith("/api/")) return;       // never cache backend calls

  if (request.mode === "navigate") {
    // Network-first for the HTML shell: always prefer a fresh load, fall
    // back to whatever was last cached only if the network is unavailable.
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match(request).then((cached) => cached || caches.match("/")))
    );
    return;
  }

  // Stale-while-revalidate for static assets (hashed JS/CSS/images/fonts):
  // serve from cache instantly if present, refresh the cache in the
  // background either way.
  event.respondWith(
    caches.match(request).then((cached) => {
      const network = fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
