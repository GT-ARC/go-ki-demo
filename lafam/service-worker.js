const CACHE_NAME = "cache-v1";
// const urlsToCache = [];

// self.addEventListener("install", (event) => {
//   event.waitUntil(
//     caches.open(CACHE_NAME).then((cache) => {
//       // console.log("Opened cache");
//       return cache.addAll(urlsToCache);
//     })
//   );
// });

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keyList => {
      return Promise.all(keyList.map(key => {
        if (key !== CACHE_NAME) {
          return caches.delete(key);
        }
      }));
    })
  );
});

self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener("fetch", (event) => {
  if (!event.request.url.startsWith("http")) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      if (response) {
        // Cache hit - return response
        return response;
      }

      // Cache miss - fetch from network
      return fetch(event.request).then((response) => {
        // Check if we received a valid response
        if (!response || response.status !== 200 || response.type !== "basic") {
          return response;
        }

        const responseClone = response.clone();
        const contentLength = response.headers.get('Content-Length');
        if (response.body && contentLength) {
          const total = parseInt(contentLength, 10);
          let loaded = 0;

          const reader = response.body.getReader();
          const stream = new ReadableStream({
            async start(controller) {
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  break;
                }
                loaded += value.byteLength;
                // Report progress
                const progress = Math.round((loaded / total) * 100);
                const fileName = event.request.url.split('/').pop();
                reportProgress(fileName, progress);
                controller.enqueue(value);
              }
              controller.close();
            }
          });

          // Update the cache with the new response
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, new Response(stream, { headers: response.headers }));
          });

          // Return the original response
          return responseClone;
        } else {
          // Update the cache
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
          
          return response;
        }
      });
    })
  );
});

function reportProgress(name, progress) {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'progress',
        name: name,
        progress: progress
      });
    });
  });
}
