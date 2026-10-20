#!/usr/bin/env python3
"""Compares the size, alignment and member offsets of every struct in src/vk.c3 with what a C compiler makes of the real headers.

    python3 -I tools/check_vk_layout.py path/to/vk.xml path/to/vulkan/include [c3c]

Prints the differences and exits 1 if there are any. VkWin32SurfaceCreateInfoKHR is not checked: it needs <windows.h>.
"""
import os
import subprocess
import sys
import tempfile

registry, include = sys.argv[1], sys.argv[2]
c3c = sys.argv[3] if len(sys.argv) > 3 else "c3c"
here = os.path.dirname(os.path.abspath(__file__))
manifest = subprocess.run([sys.executable, "-I", os.path.join(here, "gen_vk.py"), registry, "--manifest"], capture_output=True, text=True, check=True).stdout
structs = {}
constants = []
for line in manifest.splitlines():
    name, *members = line.split()
    if name == "CONSTANTS":
        constants = [m.split(":") for m in members]
    elif name != "VkWin32SurfaceCreateInfoKHR":
        structs[name] = [m.split(":") for m in members]

work = tempfile.mkdtemp()
c = ['#include <vulkan/vulkan.h>', '#include <stdio.h>', '#include <stddef.h>', 'int main(void) {']
c3 = ['module nogfx;', 'import std::io;', 'fn void main() {']
for name, members in sorted(structs.items()):
    c.append('printf("%s size %%zu align %%zu\\n", sizeof(%s), _Alignof(%s));' % (name, name, name))
    c3.append('{ %s v; io::printfn("%s size %%d align %%d", (int)@sizeof(v), (int)@alignof(v)); }' % (name, name))
    for cname, c3name in members:
        c.append('printf("%s.%s %%zu\\n", offsetof(%s, %s));' % (name, cname, name, cname))
        c3.append('{ %s v; io::printfn("%s.%s %%d", (int)((char*)&v.%s - (char*)&v)); }' % (name, name, cname, c3name))
for cname, c3name in constants:
    c.append('printf("%s %%llu\\n", (unsigned long long)(%s));' % (c3name, cname))
    c3.append('io::printfn("%s %%d", (ulong)(long)%s);' % (c3name, c3name))
c.append('return 0; }')
c3.append('}')
open(os.path.join(work, "check.c"), "w").write("\n".join(c))
open(os.path.join(work, "check.c3"), "w").write("\n".join(c3))
src = subprocess.run([sys.executable, "-I", os.path.join(here, "gen_vk.py"), registry], capture_output=True, text=True, check=True).stdout
open(os.path.join(work, "vk.c3"), "w").write(src.replace("@feat(!DARWIN, NOGFX_VULKAN)", ""))
subprocess.run(["cc", "-DVK_ENABLE_BETA_EXTENSIONS", "-I", include, "-o", os.path.join(work, "check_c"), os.path.join(work, "check.c")], check=True)
built = subprocess.run([c3c, "compile", "--output-dir", work, "-o", "check_c3", os.path.join(work, "vk.c3"), os.path.join(work, "check.c3")], capture_output=True, text=True)
if built.returncode:
    sys.exit((built.stdout + built.stderr)[:4000])
expected = subprocess.run([os.path.join(work, "check_c")], capture_output=True, text=True, check=True).stdout.splitlines()
actual = subprocess.run([os.path.join(work, "check_c3")], capture_output=True, text=True, check=True).stdout.splitlines()
bad = [(e, a) for e, a in zip(expected, actual) if e != a]
if len(expected) != len(actual):
    print("line count differs: C %d, C3 %d" % (len(expected), len(actual)))
for e, a in bad:
    print("C: %s    C3: %s" % (e, a))
print("%d structs, %d constants, %d values checked, %d differences" % (len(structs), len(constants), len(expected), len(bad) + (len(expected) != len(actual))))
sys.exit(1 if bad or len(expected) != len(actual) else 0)
