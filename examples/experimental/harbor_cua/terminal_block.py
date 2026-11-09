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

# LIN-2752's detectors for a shell command or a line of code in typed text. Command words count only in command
# position, since "Introduction to Cells - ..." is prose; DBeaver's ".shell"/"!cmd" and Basic macros are shells too.
_SHELL = re.compile(
    r"(cat > |cat <<|blender -b|ogr2ogr|find / |cd /home|<< .EOF|\.shell |bash -c|\bShell\s*\(?\x22|^\s*Sub \w+\s*\("
    r"|^!\w|(?:^|\s)(?:python3|mkdir|sudo) |(?:^[ \t]*|[;&|(]\s*)(?:ls -|echo |find |ls |cd |sed |grep |cp |mv |chmod |head |pip |cat ))",
    re.M,
)
_CODE = re.compile(r"(^import |\bimport \w|print\(|slicer\.|bpy\.|\bdef |\w+ = \w+\.\w+\()")
# An app's own file typed out whole (draw.io/SUMO/VLC XML, Webots VRML, a GMAT script) skips its GUI like `cat > file`.
_APP_FILE = re.compile(
    r"#VRML|#!WRML|<\?xml|^\s*<\w+[\s>/]|^\s*\w+\s*\{\s*$|\bGMATScript\b|\bBEGINScript\b|\bBeginMissionSequence\b"
    r"|\bPropagate\s+\w+\(|\.(?:SMA|ECC|Epoch|DateFormat)\s*=|^\s*Begin\s+([A-Z]\w*)\s*$[\s\S]*?^\s*End\s+\1\s*$"
    r"|^\s*Create\s+(?:Spacecraft|Propagator|ForceModel|ImpulsiveBurn|FiniteBurn|ReportFile|XYPlot|OrbitView|GroundTrackPlot"
    r"|Variable|Array|String|CoordinateSystem|GroundStation|DifferentialCorrector|Formation|Thruster|ChemicalTank|FuelTank"
    r"|EphemerisFile|Barycenter|LibrationPoint|Optimizer)\b",
    re.M,
)

# One Webots node typed whole ("Fog { fogType ... }") is under the length bar below, so a valued node counts alone.
_VRML_NODE = re.compile(
    r"(?:^|[\s\[])(?:DEF\s+\w+\s+)?[A-Z][A-Za-z0-9]*\s*\{\s*[a-z][A-Za-z0-9]*\s+"
    r"(?:-?\.?\d|\x22|\[|TRUE\b|FALSE\b|[A-Z]\w*\s*\{)[^{}]*\}"
)


def typed_code(text: str) -> bool:
    if _SHELL.search(text) or _CODE.search(text) or _VRML_NODE.search(text):
        return True
    return len(text) > 150 and text.count("\n") >= 4 and bool(_APP_FILE.search(text))
