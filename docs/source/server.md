# Running the app on a server

`pypalc gui` (see {doc}`gui`) can serve the app to other computers. The
app needs a websocket connection between the browser and the server, which
Panel (Bokeh) only accepts from allowed hosts: `--allow-websocket-origin`
must name the host that users type into their browser, the public name of
the server, not the address the server listens on.

Without `--address`, the server listens on all network interfaces of the
computer; browsers on other hosts still get no app unless their host is
allowed with `--allow-websocket-origin` (by default only
`localhost:PORT`). `--address` restricts the interface, for example to
`127.0.0.1` behind a reverse proxy.

The configurations below are examples to adapt: replace the host names,
ports, paths and the user with your own.

## Directly

Install pyPALC with the GUI on the server (for example in a virtual
environment, `python3 -m venv /opt/pypalc/venv` and
`/opt/pypalc/venv/bin/pip install "pypalc[gui]"`) and start it:

```console
pypalc gui --address 0.0.0.0 --port 5006 --allow-websocket-origin pypalc.example.org:5006 --no-show
```

The app is then at `http://pypalc.example.org:5006/pyPALC`. The connection
is not encrypted; for a public server, put a web server with HTTPS in
front of it.

## Behind Apache

A reverse proxy forwards the requests of the browsers to the Panel server,
which then only needs to listen on the local address
(`--address 127.0.0.1`), so that its port cannot be reached from other
computers. Apache needs
the modules `mod_proxy`, `mod_proxy_http` and `mod_proxy_wstunnel` (on
Debian and Ubuntu: `sudo a2enmod proxy proxy_http proxy_wstunnel`). The
app's websocket is at the path `/pyPALC/ws` below the app's prefix; it
needs its own `ProxyPass` rule with `ws://`, placed before the rule for
the other requests, because Apache uses the first rule that matches.

### A whole host

The app is the only content of `pypalc.example.org`, at
`https://pypalc.example.org/pyPALC`.

```console
pypalc gui --address 127.0.0.1 --port 5006 --allow-websocket-origin pypalc.example.org --no-show
```

```apache
<VirtualHost *:443>
    ServerName pypalc.example.org

    # SSLEngine on, SSLCertificateFile, SSLCertificateKeyFile ...

    ProxyPreserveHost On
    ProxyPass        /pyPALC/ws ws://127.0.0.1:5006/pyPALC/ws
    ProxyPassReverse /pyPALC/ws ws://127.0.0.1:5006/pyPALC/ws
    ProxyPass        / http://127.0.0.1:5006/
    ProxyPassReverse / http://127.0.0.1:5006/
</VirtualHost>
```

### A sub-path

The app shares the host `www.example.org` with other content and is at
`https://www.example.org/pypalc/pyPALC`. `--prefix /pypalc` makes the
Panel server serve everything (the app, its websocket and its static
files) below `/pypalc`, so that Apache can forward that path unchanged:

```console
pypalc gui --address 127.0.0.1 --port 5006 --prefix /pypalc --allow-websocket-origin www.example.org --no-show
```

```apache
<VirtualHost *:443>
    ServerName www.example.org

    # SSLEngine on, SSLCertificateFile, SSLCertificateKeyFile ...
    # ... the other content of the host ...

    ProxyPreserveHost On
    ProxyPass        /pypalc/pyPALC/ws ws://127.0.0.1:5006/pypalc/pyPALC/ws
    ProxyPassReverse /pypalc/pyPALC/ws ws://127.0.0.1:5006/pypalc/pyPALC/ws
    ProxyPass        /pypalc/ http://127.0.0.1:5006/pypalc/
    ProxyPassReverse /pypalc/ http://127.0.0.1:5006/pypalc/
</VirtualHost>
```

The browser opens the websocket with `wss://` on an HTTPS page; Apache
ends the encryption and forwards it as `ws://` to the Panel server.

After a change of the configuration, check it with `sudo apachectl
configtest` and reload Apache (`sudo systemctl reload apache2`).

## As a systemd service

A systemd unit starts the server at boot and restarts it after a failure,
here for the sub-path setup above, with pyPALC installed in
`/opt/pypalc/venv` and run by the user `pypalc`. Save it as
`/etc/systemd/system/pypalc.service`:

```ini
[Unit]
Description=pyPALC (Panel server)
After=network.target

[Service]
Type=simple
User=pypalc
Environment=PYTHONNOUSERSITE=1
ExecStart=/opt/pypalc/venv/bin/pypalc gui --address 127.0.0.1 --port 5006 --prefix /pypalc --allow-websocket-origin www.example.org --no-show
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then enable and start it, and read its log:

```console
sudo systemctl daemon-reload
sudo systemctl enable --now pypalc
journalctl -u pypalc
```

## From a checkout

`panel serve` takes the same options, for the app in a clone of the
repository (run from the folder that contains the `pyPALC` folder):

```console
panel serve pyPALC --address 127.0.0.1 --port 5006 --prefix /pypalc --allow-websocket-origin www.example.org
```
