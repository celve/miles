"""Take a GNOME desktop's shells away from its user before an episode, and recognise shell or code typed anyway.

CUA-World's tasks can all be done from a terminal, and a policy that may use one learns the terminal rather than the
GUI (LIN-2752: 143 of 144 passes of a trained Qwen3.8-27B came from shell work). The block runs as root after the task's
own setup; the in-app consoles it cannot remove (Blender's, QGIS's, Slicer's Python, PyMOL's command line) are what
``typed_code`` is for.
"""

import re

# Every terminal emulator and VS Code (its integrated terminal is a shell) become root-only, their launcher entries
# go, the warm terminal server dies, and Ctrl+Alt+T and the Alt+F2 run dialog are switched off.
BLOCK_SCRIPT = r"""
set -u
U=$1
blocked=""
for name in gnome-terminal gnome-terminal.real gnome-terminal.wrapper gnome-terminal-server x-terminal-emulator xterm uxterm \
    lxterm koi8rxterm xfce4-terminal lxterminal terminator tilix konsole kgx gnome-console mate-terminal urxvt rxvt \
    alacritty kitty sakura guake tilda xdg-terminal-exec code; do
  for p in $(which -a "$name" 2>/dev/null) /usr/bin/$name /usr/local/bin/$name /usr/libexec/$name /usr/lib/gnome-terminal/$name; do
    if [ -f "$p" ] && [ "$(stat -c %a "$p")" != 700 ]; then chown root:root "$p"; chmod 0700 "$p"; blocked="$blocked $p"; fi
  done
done
for p in /usr/share/code/code /usr/share/code/bin/code; do [ -f "$p" ] && chown root:root "$p" && chmod 0700 "$p" && blocked="$blocked $p"; done
for f in /usr/share/applications/*erminal*.desktop /usr/share/applications/*xterm*.desktop /usr/share/applications/code*.desktop \
    /usr/share/applications/org.gnome.Console.desktop /home/$U/.local/share/applications/*erminal*.desktop; do
  [ -e "$f" ] && mv "$f" "$f.blocked"
done
pkill -f '^bash -c /usr/local/bin/warm-terminal'; pkill -x gnome-terminal-
gs() { runuser -u $U -- env HOME=/home/$U DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u $U)/bus" gsettings "$@"; }
gs set org.gnome.settings-daemon.plugins.media-keys terminal "[]"
gs set org.gnome.desktop.lockdown disable-command-line true
sleep 1
# envd never reaps, so a killed terminal server can linger as a zombie, which is no terminal.
ps -eo stat=,comm= | awk '$1 !~ /^Z/ && $2 == "gnome-terminal-"' | grep -q . && echo "STILL-RUNNABLE gnome-terminal-server"
su $U -s /bin/sh -c "test -x /usr/bin/gnome-terminal" && echo "STILL-RUNNABLE gnome-terminal"
echo "blocked:$blocked"
"""

# LIN-2752's detectors for a shell command or a line of code in typed text.
_SHELL = re.compile(
    r"(cat > |cat <<|python3 |ls -|mkdir |blender -b|ogr2ogr|echo |find / |cd /home|<< .EOF"
    r"|^find |^ls |^cd |^sed |^grep |^cp |^mv |^chmod |sudo |^head |^pip |^cat )"
)
_CODE = re.compile(r"(^import |\bimport \w|print\(|slicer\.|bpy\.|\bdef |\w+ = \w+\.\w+\()")


def typed_code(text: str) -> bool:
    return bool(_SHELL.search(text) or _CODE.search(text))
