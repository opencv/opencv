#!/bin/bash

groupadd -r appgroup -g $APP_GID
useradd -u $APP_UID -r -g appgroup -d /home/appuser -m -s /bin/bash -c "App user" appuser

exec su - appuser -c "cd /app && ./make_mingw.sh /build"
