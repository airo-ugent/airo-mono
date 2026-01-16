"""
This package implements a shared-memory camera proxy.

Publisher:
- Owns real camera
- Serializes camera outputs into shared memory buffers

Receiver:
- Deserializes shared memory buffers
- Exposes standard Camera interfaces via mixins
"""
