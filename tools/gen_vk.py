#!/usr/bin/env python3
"""Generates src/vk.c3, the part of Vulkan that the backend talks to, from the Khronos registry (vk.xml).

    python3 -I tools/gen_vk.py path/to/Vulkan-Headers/registry/vk.xml > src/vk.c3

It emits the commands in COMMANDS, the structs that they and STRUCTS need (with the structs those contain), the enums those use, and the
constants that the backend refers to. The layouts are the registry's, so there are no hand written structs to get wrong. Needs a registry
with VK_KHR_device_address_commands (Vulkan headers 1.4.357 or later), which is what upstream requires too.
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

# Commands that the backend calls. The first parameter decides where it is loaded from, see level().
COMMANDS = """
vkCreateInstance vkEnumerateInstanceVersion vkEnumerateInstanceExtensionProperties vkEnumerateInstanceLayerProperties
vkDestroyInstance vkEnumeratePhysicalDevices vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceMemoryProperties
vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceImageFormatProperties2
vkEnumerateDeviceExtensionProperties vkCreateDevice vkGetDeviceProcAddr vkCreateDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT
vkDestroySurfaceKHR vkCreateWin32SurfaceKHR vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR
vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceFormatsKHR
vkDestroyDevice vkGetDeviceQueue vkCreateBuffer vkDestroyBuffer vkAllocateMemory vkFreeMemory vkBindBufferMemory vkMapMemory vkUnmapMemory
vkGetBufferDeviceAddress vkGetDeviceBufferMemoryRequirements vkGetDeviceImageMemoryRequirements vkGetBufferMemoryRequirements vkCreateImage vkDestroyImage vkBindImageMemory
vkCreateImageView vkDestroyImageView vkCreateSemaphore vkDestroySemaphore vkGetSemaphoreCounterValue vkWaitSemaphores vkCreateFence
vkDestroyFence vkWaitForFences vkResetFences vkGetFenceStatus vkCreateCommandPool vkDestroyCommandPool vkResetCommandPool
vkAllocateCommandBuffers vkFreeCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer vkQueueSubmit2 vkQueuePresentKHR vkDeviceWaitIdle
vkCreateQueryPool vkDestroyQueryPool vkResetQueryPool vkGetQueryPoolResults vkCmdWriteTimestamp2 vkCmdPipelineBarrier2
vkCreateGraphicsPipelines vkCreateComputePipelines vkDestroyPipeline vkCmdBindPipeline vkCmdSetViewportWithCount vkCmdSetScissorWithCount
vkCmdSetDepthTestEnable vkCmdSetDepthWriteEnable vkCmdSetDepthCompareOp vkCmdSetStencilTestEnable vkCmdSetStencilOp
vkCmdSetStencilCompareMask vkCmdSetStencilWriteMask vkCmdSetStencilReference vkCmdBeginRendering vkCmdEndRendering vkCmdDraw
vkCmdDrawIndexed vkCmdDispatch vkCreateSwapchainKHR vkDestroySwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR
vkWriteSamplerDescriptorsEXT vkWriteResourceDescriptorsEXT vkCmdBindSamplerHeapEXT vkCmdBindResourceHeapEXT vkCmdPushDataEXT
vkCmdBindIndexBuffer3KHR vkCmdDrawIndirect2KHR vkCmdDrawIndexedIndirect2KHR vkCmdDispatchIndirect2KHR vkCmdCopyMemoryKHR
vkCmdCopyMemoryToImageKHR vkCmdCopyImageToMemoryKHR vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksIndirect2EXT
vkGetPhysicalDeviceProperties vkCmdBindIndexBuffer2 vkCmdDrawIndirect vkCmdDrawIndexedIndirect vkCmdDispatchIndirect vkCmdDrawMeshTasksIndirectEXT
vkCmdCopyBuffer2 vkCmdCopyBufferToImage2 vkCmdCopyImageToBuffer2
""".split()

# Structs that are only reached through a pNext chain or a union, so no command mentions them.
STRUCTS = """
VkPhysicalDeviceVulkan11Features VkPhysicalDeviceVulkan12Features VkPhysicalDeviceVulkan13Features VkPhysicalDeviceVulkan14Features
VkPhysicalDeviceVulkan12Properties VkPhysicalDeviceDescriptorHeapFeaturesEXT VkPhysicalDeviceDescriptorHeapPropertiesEXT
VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR VkPhysicalDeviceShaderUntypedPointersFeaturesKHR
VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR VkPhysicalDeviceMeshShaderFeaturesEXT VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR
VkFormatProperties3 VkImageFormatListCreateInfo VkImageViewUsageCreateInfo VkMemoryAllocateFlagsInfo VkSemaphoreTypeCreateInfo
VkPipelineRenderingCreateInfo VkPipelineCreateFlags2CreateInfo VkShaderDescriptorSetAndBindingMappingInfoEXT VkShaderModuleCreateInfo
VkSurfacePresentModeKHR VkSwapchainPresentModesCreateInfoKHR VkSwapchainPresentFenceInfoKHR VkWin32SurfaceCreateInfoKHR
VkDebugUtilsMessengerCreateInfoEXT VkDebugUtilsMessengerCallbackDataEXT
VkDrawIndirectCommand VkDrawIndexedIndirectCommand VkDrawMeshTasksIndirectCommandEXT
""".split()

# Extensions whose enum values, constants and names are emitted besides the core versions.
EXTENSIONS = """
VK_KHR_surface VK_KHR_swapchain VK_KHR_win32_surface VK_KHR_get_surface_capabilities2 VK_KHR_surface_maintenance1
VK_KHR_swapchain_maintenance1 VK_EXT_surface_maintenance1 VK_EXT_swapchain_maintenance1 VK_EXT_swapchain_colorspace VK_EXT_debug_utils
VK_EXT_descriptor_heap VK_KHR_device_address_commands VK_KHR_shader_untyped_pointers VK_KHR_unified_image_layouts VK_EXT_mesh_shader
VK_KHR_maintenance5 VK_AMD_device_coherent_memory VK_QCOM_tile_memory_heap
""".split()

# C3 wants constants all upper case, so VK_FORMAT_ASTC_4x4_UNORM_BLOCK is VK_FORMAT_ASTC_4X4_UNORM_BLOCK here.

# Union members that point at types from extensions the backend does not use. Every member of a union is a pointer here, so the layout is the same.
SKIP_MEMBERS = {("VkResourceDescriptorDataEXT", "pTensorARM")}

C_TYPES = {
    "void": "void", "char": "char", "int": "int", "float": "float", "double": "double", "uint8_t": "char", "int8_t": "ichar",
    "uint16_t": "ushort", "int16_t": "short", "uint32_t": "uint", "int32_t": "int", "uint64_t": "ulong", "int64_t": "long",
    "size_t": "usz",
    # Platform types: only the Win32 surface mentions them.
    "HINSTANCE": "void*", "HWND": "void*", "HMONITOR": "void*", "HANDLE": "void*", "DWORD": "uint", "LPCWSTR": "void*",
    "SECURITY_ATTRIBUTES": "void", "PFN_vkVoidFunction": "PFN_vkVoidFunction",
}

C3_KEYWORDS = {"module", "default", "fn", "var", "int", "char", "const", "struct", "return", "case", "switch", "alias", "enum", "union", "import",
                "macro", "extern", "static", "defer", "try", "catch", "assert", "tlocal", "inline", "bitstruct", "faultdef", "typedef", "any",
                "true", "false", "null", "void", "bool", "float", "double", "long", "short", "ulong", "uint", "ichar", "usz", "isz", "if",
                "else", "for", "foreach", "while", "do", "break", "continue", "nextcase", "asm"}


def is_vulkan(e):
    """Not an element that only exists in the Vulkan SC profile."""
    api = e.get("api")
    return api is None or "vulkan" in api.split(",")


def die(message):
    sys.exit("gen_vk: " + message)


class Registry:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.types = {}      # name -> element, aliases resolved to the target
        self.aliases = {}
        self.enums = {}      # enum type name -> {"kind": "enum"/"bitmask", "bits": 32/64, "values": {name: int}}
        self.constants = {}  # API constants
        self.extension_names = {}
        self.commands = {}
        self.vendor_values = {}   # every enum value that any extension adds, by name, so that aliases into extensions we do not emit resolve
        self.vendor_alias = {}
        self.header_version = None
        self.read_types()
        self.read_enums()
        self.read_commands()
        self.read_features()

    def read_types(self):
        for ty in self.root.find("types"):
            if not is_vulkan(ty):
                continue
            name = ty.get("name")
            if name is None:
                n = ty.find("name")
                name = n.text if n is not None else None
            if name is None:
                continue
            if ty.get("alias"):
                self.aliases[name] = ty.get("alias")
                continue
            self.types[name] = ty
            if ty.get("category") == "define" and name == "VK_HEADER_VERSION":
                self.header_version = "".join(ty.itertext()).split()[-1]

    def resolve(self, name):
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def read_enums(self):
        for block in self.root.findall("enums"):
            name = block.get("name")
            if name == "API Constants":
                for e in block.findall("enum"):
                    self.constants[e.get("name")] = e
                continue
            kind = block.get("type")
            if kind not in ("enum", "bitmask"):
                continue
            info = {"kind": kind, "bits": int(block.get("bitwidth", "32")), "values": {}, "alias": {}}
            self.enums[name] = info
            for e in block.findall("enum"):
                if is_vulkan(e):
                    self.add_enum_value(info, e, None)

    def value_of(self, e, extnumber):
        if e.get("bitpos") is not None:
            return 1 << int(e.get("bitpos"))
        if e.get("offset") is not None:
            number = int(e.get("extnumber") or extnumber)
            v = 1000000000 + (number - 1) * 1000 + int(e.get("offset"))
            return -v if e.get("dir") == "-" else v
        if e.get("value") is not None:
            text = e.get("value")
            try:
                return int(text, 0)
            except ValueError:
                return text
        return None

    def add_enum_value(self, info, e, extnumber):
        name = e.get("name")
        if "_RESERVED_" in name:
            return   # Bits that the registry reserves for later and the C headers do not define.
        if e.get("alias"):
            info["alias"][name] = e.get("alias")
            return
        v = self.value_of(e, extnumber)
        if v is not None:
            info["values"][name] = v

    def read_commands(self):
        for c in self.root.find("commands"):
            if not is_vulkan(c):
                continue
            proto = c.find("proto")
            if proto is None:
                if c.get("alias"):
                    self.aliases[c.get("name")] = c.get("alias")
                continue
            self.commands[proto.find("name").text] = c

    def read_features(self):
        wanted = set(EXTENSIONS)
        for feature in self.root.findall("feature"):
            if "vulkan" not in feature.get("api", "").split(","):
                continue
            for req in feature.findall("require"):
                self.read_require(req, None)
        for ext in self.root.find("extensions"):
            number = ext.get("number")
            for req in ext.findall("require"):
                # Besides its own extension, a require block that depends on one of ours: VK_KHR_synchronization2 adds the
                # task and mesh stages to its stage flags for VK_EXT_mesh_shader.
                depends = set(re.findall(r"\w+", req.get("depends", "")))
                if ext.get("name") in wanted or depends & wanted:
                    self.read_require(req, number)
                for e in req.findall("enum"):
                    if not is_vulkan(e) or not e.get("name"):
                        continue
                    if e.get("alias"):
                        self.vendor_alias[e.get("name")] = e.get("alias")
                    else:
                        v = self.value_of(e, number)
                        if v is not None:
                            self.vendor_values[e.get("name")] = v

    def read_require(self, req, extnumber):
        for e in req.findall("enum"):
            if not is_vulkan(e):
                continue
            extends = e.get("extends")
            name = e.get("name")
            if extends:
                info = self.enums.get(extends)
                if info is None:
                    continue
                self.add_enum_value(info, e, extnumber)
            elif name.endswith("_EXTENSION_NAME"):
                self.extension_names[name] = e.get("value").strip('"').replace("&quot;", "")

    def enum_value(self, info, name):
        for _ in range(8):
            if name in info["values"]:
                return info["values"][name]
            if name in info["alias"]:
                name = info["alias"][name]
            elif name in self.vendor_alias:
                name = self.vendor_alias[name]
            else:
                break
        return info["values"].get(name, self.vendor_values.get(name))


reg = None


class Decl:
    """A member or parameter: C type text, pointer depth, the name and any array dimensions."""

    def __init__(self, element):
        self.pre = element.text or ""
        self.type = None
        self.name = None
        self.post_type = ""
        self.dims = []
        self.bits = None
        tail = ""
        state = "pre"
        pieces = []
        for ch in element:
            if ch.tag == "type":
                self.type = ch.text
                state = "post_type"
                self.post_type += ch.tail or ""
            elif ch.tag == "name":
                self.name = ch.text
                state = "post_name"
                pieces.append(ch.tail or "")
            elif ch.tag == "enum":
                pieces.append(("enum", ch.text))
                pieces.append(ch.tail or "")
            elif ch.tag == "comment":
                if state == "post_name":
                    pieces.append(ch.tail or "")
                elif state == "post_type":
                    self.post_type += ch.tail or ""
        # Array dimensions: [4], [VK_MAX_X], [2][3]
        text = ""
        for p in pieces:
            text += p if isinstance(p, str) else "\x00" + p[1] + "\x01"
        for m in re.finditer(r"\[([^\]]*)\]", text):
            dim = m.group(1)
            if dim.startswith("\x00"):
                dim = dim[1:-1]
            self.dims.append(dim)
        m = re.search(r":\s*(\d+)", text)
        if m:
            self.bits = int(m.group(1))
        self.pointers = self.post_type.count("*")
        self.is_const = "const" in self.pre


def c3_name(name):
    return name + "_" if name in C3_KEYWORDS else name


def handle_kinds():
    result = {}
    for name, ty in reg.types.items():
        if ty.get("category") == "handle":
            inner = ty.find("type")
            result[name] = "void*" if inner is not None and inner.text == "VK_DEFINE_HANDLE" else "ulong"
    return result


HANDLES = None
BASETYPES = {}
BITMASKS = {}


def base_c3(ctype):
    """The C3 spelling of a registry type name, resolving aliases. Records the dependency."""
    ctype = reg.resolve(ctype)
    if ctype in C_TYPES:
        return C_TYPES[ctype]
    return ctype


class Emitter:
    def __init__(self):
        self.struct_order = []
        self.needed_structs = set()
        self.needed_enums = set()
        self.needed_handles = set()
        self.needed_basetypes = set()
        self.needed_bitmasks = set()
        self.needed_pfns = set()
        self.queue = []

    def use_type(self, ctype):
        ctype = reg.resolve(ctype)
        if ctype in C_TYPES:
            return
        ty = reg.types.get(ctype)
        if ty is None:
            if ctype.startswith("PFN_"):
                return
            die("unknown type " + ctype)
        category = ty.get("category")
        if category in ("struct", "union"):
            if ctype not in self.needed_structs:
                self.needed_structs.add(ctype)
                self.queue.append(ctype)
        elif category == "enum":
            self.needed_enums.add(ctype)
        elif category == "handle":
            self.needed_handles.add(ctype)
        elif category == "basetype":
            self.needed_basetypes.add(ctype)
        elif category == "bitmask":
            self.needed_bitmasks.add(ctype)
            bits = reg.resolve(ty.get("requires") or ty.get("bitvalues") or "")
            if bits in reg.enums:
                self.needed_enums.add(bits)
        elif category == "funcpointer":
            self.needed_pfns.add(ctype)
        else:
            die("unhandled category %s for %s" % (category, ctype))

    def c3_type(self, d, param=False):
        base = base_c3(d.type)
        self.use_type(d.type)
        depth = d.pointers
        if base == "PFN_vkVoidFunction":
            self.needed_pfns.add("PFN_vkVoidFunction")
        elif base.startswith("PFN_vk") and d.type not in ("PFN_vkDebugUtilsMessengerCallbackEXT", "PFN_vkVoidFunction"):
            base = "void*"
        elif base.startswith("PFN_vk"):
            self.needed_pfns.add(base)
        if d.type == "void" and depth == 0:
            text = "void"
        else:
            text = base
        if base == "char" and depth >= 1:
            text = "ZString"
            depth -= 1
        text += "*" * depth
        if param and d.dims:
            text += "*"
        return text

    def struct_text(self, name):
        ty = reg.types[name]
        union = ty.get("category") == "union"
        lines = ["%s %s" % ("union" if union else "struct", name), "{"]
        for m in ty.findall("member"):
            if not is_vulkan(m):
                continue
            d = Decl(m)
            if d.bits is not None:
                die("bitfield in " + name)
            if d.type is None or d.name is None or (name, d.name) in SKIP_MEMBERS:
                continue
            ctype = self.c3_type(d)
            for dim in reversed(d.dims):
                size = dim
                if not size.isdigit():
                    size = reg.constants[size].get("value") if size in reg.constants else die("dimension " + size)
                    size = size.strip("()U")
                ctype += "[%s]" % size
            if m.find("name").text == "sType" and m.get("values"):
                reg_stypes_needed.add(m.get("values").split(",")[0])
            lines.append("\t%s %s;" % (ctype, c3_name(d.name)))
            MANIFEST.setdefault(name, []).append((d.name, c3_name(d.name)))
        lines.append("}")
        return "\n".join(lines)


reg_stypes_needed = set()
CONSTS = []     # (registry name, C3 name) of every enum value and structure type
MANIFEST = {}   # struct -> [(registry member name, C3 member name)], for tools/check_vk_layout.py


def command_signature(em, cmd):
    proto = cmd.find("proto")
    ret = Decl(proto)
    rtype = em.c3_type(ret)
    params = []
    for p in cmd.findall("param"):
        if p.get("api") and "vulkan" not in p.get("api").split(","):
            continue
        d = Decl(p)
        params.append((c3_name(d.name), em.c3_type(d, param=True), d))
    return rtype, params


def level(name, params):
    if name == "vkGetDeviceProcAddr":
        return "instance"
    if not params:
        return "global"
    first = params[0][2].type
    if first in ("VkInstance", "VkPhysicalDevice"):
        return "instance"
    if first in ("VkDevice", "VkQueue", "VkCommandBuffer"):
        return "device"
    return "global"


API_CONSTANTS = """VK_MAX_PHYSICAL_DEVICE_NAME_SIZE VK_UUID_SIZE VK_LUID_SIZE VK_MAX_EXTENSION_NAME_SIZE VK_MAX_DESCRIPTION_SIZE VK_MAX_MEMORY_TYPES
VK_MAX_MEMORY_HEAPS VK_LOD_CLAMP_NONE VK_REMAINING_MIP_LEVELS VK_REMAINING_ARRAY_LAYERS VK_WHOLE_SIZE VK_ATTACHMENT_UNUSED VK_TRUE VK_FALSE
VK_QUEUE_FAMILY_IGNORED VK_MAX_DRIVER_NAME_SIZE VK_MAX_DRIVER_INFO_SIZE VK_MAX_DEVICE_GROUP_SIZE""".split()


def used_names():
    """The VK_ names that the backend, the tests and the example mention: the constants that are emitted. A name that is used and missing
    fails to compile, and then this is run again."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = set()
    for pattern in ("src/*.c3", "tests/*.c3", "tests/*/*.c3", "examples/*.c3"):
        for path in glob.glob(os.path.join(root, pattern)):
            if os.path.basename(path) != "vk.c3":
                names |= set(re.findall(r"\bVK_[A-Z0-9_]+\b", open(path).read()))
    return names


def parse_constant(text, kind):
    text = text.strip()
    if kind == "float":
        return "%s" % float(text.rstrip("fF"))
    m = re.fullmatch(r"\(~(\d+)(U|ULL)\)", text)
    if m:
        bits = 64 if m.group(2) == "ULL" else 32
        return str(((1 << bits) - 1) - int(m.group(1)))
    if re.fullmatch(r"\d+", text):
        return text
    die("constant " + text)


def main():
    global reg, HANDLES
    reg = Registry(sys.argv[1])
    used = used_names()
    HANDLES = handle_kinds()
    em = Emitter()
    out = []

    # Commands first: they decide which types are needed.
    commands = []
    for name in COMMANDS:
        cmd = reg.commands.get(name)
        if cmd is None:
            die("no command " + name)
        rtype, params = command_signature(em, cmd)
        commands.append((name, rtype, params))
    for name in STRUCTS:
        em.use_type(name)
    structs_text = {}
    while em.queue:
        name = em.queue.pop()
        structs_text[name] = em.struct_text(name)

    # Enum types reached by struct members and parameters.
    enum_lines = []
    for name in sorted(em.needed_enums):
        info = reg.enums.get(name)
        if info is None:
            die("enum without values " + name)
        c3 = ("ulong" if info["bits"] == 64 else "uint") if info["kind"] == "bitmask" else "int"
        enum_lines.append("alias %s = %s;" % (name, c3))
        names = sorted(set(info["values"]) | set(info["alias"]))
        for n in names:
            if name == "VkStructureType":
                continue
            v = reg.enum_value(info, n)
            if v is None or isinstance(v, str):
                continue
            if info["kind"] == "bitmask" and n.endswith("_FLAG_BITS_MAX_ENUM"):
                continue
            if n.endswith("_MAX_ENUM") or n.upper() not in used:
                continue
            enum_lines.append("const %s %s = %d;" % (name, n.upper(), v))
            CONSTS.append((n, n.upper()))
    stype_info = reg.enums["VkStructureType"]
    stype_lines = []
    for n in sorted(reg_stypes_needed):
        v = reg.enum_value(stype_info, n)
        if not isinstance(v, int):
            die("no sType value for " + n)
        if n.upper() not in used:
            continue
        stype_lines.append("const VkStructureType %s = %d;" % (n.upper(), v))
        CONSTS.append((n, n.upper()))

    out.append("// Generated by tools/gen_vk.py from the Khronos Vulkan registry (VK_HEADER_VERSION %s). Do not edit." % reg.header_version)
    out.append("//")
    out.append("// The part of Vulkan that the backend uses. Struct layouts are the registry's. Handles are distinct types, enums and flags are plain")
    out.append("// integers. There is no link time dependency on Vulkan: the loader is opened at run time and the commands are looked up, see VkFns.")
    out.append("module nogfx @feat(!DARWIN, NOGFX_VULKAN);")
    out.append("")
    out.append("const uint VK_API_VERSION_1_0 = 0;")
    out.append("const uint VK_API_VERSION_1_4 = (1 << 22) | (4 << 12);")
    out.append("fn uint vk_make_api_version(uint variant, uint major, uint minor, uint patch) => (variant << 29) | (major << 22) | (minor << 12) | patch;")
    out.append("")
    out.append("// ---- Handles " + "-" * 100)
    out.append("")
    for name in sorted(em.needed_handles):
        out.append("typedef %s @constinit = %s;" % (name, HANDLES[name]))
    out.append("")
    out.append("// ---- Base types and flags " + "-" * 87)
    out.append("")
    for name in sorted(em.needed_basetypes):
        ty = reg.types[name]
        inner = ty.find("type")
        out.append("alias %s = %s;" % (name, C_TYPES[inner.text]))
    for name in sorted(em.needed_bitmasks):
        ty = reg.types[name]
        inner = ty.find("type").text
        out.append("alias %s = %s;" % (name, "ulong" if inner == "VkFlags64" else "uint"))
    out.append("")
    out.append("// ---- API constants " + "-" * 94)
    out.append("")
    for name in API_CONSTANTS:
        if name not in used:
            continue
        e = reg.constants[name]
        kind = e.get("type")
        out.append("const %s %s = %s;" % ({"float": "float", "uint64_t": "ulong"}.get(kind, "uint"), name, parse_constant(e.get("value"), kind)))
    out.append("")
    out.append("// ---- Extension names " + "-" * 92)
    out.append("")
    for name, text in sorted(reg.extension_names.items()):
        if name not in used:
            continue
        out.append('const ZString %s = "%s";' % (name, text))
    out.append("")
    out.append("// ---- Enums and flag bits " + "-" * 88)
    out.append("")
    out.extend(enum_lines)
    out.append("")
    out.extend(stype_lines)
    out.append("")
    out.append("// ---- Structs " + "-" * 100)
    out.append("")
    for name in sorted(structs_text):
        out.append(structs_text[name])
        out.append("")
    out.append("// ---- Function pointers " + "-" * 90)
    out.append("")
    out.append("alias PFN_vkVoidFunction = fn void();")
    if "PFN_vkDebugUtilsMessengerCallbackEXT" in em.needed_pfns:
        out.append("alias PFN_vkDebugUtilsMessengerCallbackEXT = fn VkBool32(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, "
                   "VkDebugUtilsMessengerCallbackDataEXT*, void*);")
    out.append("alias PFN_vkGetInstanceProcAddr = fn PFN_vkVoidFunction(VkInstance, ZString);")
    for name, rtype, params in commands:
        out.append("alias PFN_%s = fn %s(%s);" % (name, rtype, ", ".join("%s %s" % (t, n) for n, t, _ in params)))
    out.append("")
    out.append("// One table per device: the commands are looked up through the instance or the device, so every call goes straight to the driver.")
    out.append("struct VkFns")
    out.append("{")
    out.append("\tPFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;")
    for name, rtype, params in commands:
        out.append("\tPFN_%s %s;" % (name, name))
    out.append("}")
    out.append("")
    for lvl, argline in (("global", ""), ("instance", "VkInstance instance"), ("device", "VkDevice device")):
        names = [c for c in commands if level(c[0], c[2]) == lvl]
        if lvl == "device":
            out.append("fn void VkFns.load_%s(&self, %s, PFN_vkGetDeviceProcAddr get)" % (lvl, argline))
        else:
            out.append("fn void VkFns.load_%s(&self%s)" % (lvl, (", " + argline) if argline else ""))
        out.append("{")
        for name, rtype, params in names:
            if lvl == "global":
                out.append("\tself.%s = (PFN_%s)self.vkGetInstanceProcAddr((VkInstance)null, \"%s\");" % (name, name, name))
            elif lvl == "instance":
                out.append("\tself.%s = (PFN_%s)self.vkGetInstanceProcAddr(instance, \"%s\");" % (name, name, name))
            else:
                out.append("\tself.%s = (PFN_%s)get(device, \"%s\");" % (name, name, name))
        out.append("}")
        out.append("")
    if "--manifest" in sys.argv:
        for name in sorted(MANIFEST):
            print(name, *[m[0] + ":" + m[1] for m in MANIFEST[name]])
        print("CONSTANTS", *[c[0] + ":" + c[1] for c in CONSTS])
        return
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
