# Reference Lua interpreter (no LuaJIT).
FROM alpine:3.20
RUN apk add --no-cache lua5.4
