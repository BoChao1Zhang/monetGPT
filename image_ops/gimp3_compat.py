"""
GIMP 3.0 GObject Introspection API compatibility helpers.

Provides a clean calling interface for GIMP 3.0 PDB procedures,
replacing the old gimpfu/pdb.* calling convention from GIMP 2.10.
"""
import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp, Gio


def call_pdb(proc_name, **props):
    """Call a GIMP PDB procedure with named arguments."""
    pdb = Gimp.get_pdb()
    proc = pdb.lookup_procedure(proc_name)
    config = proc.create_config()
    for key, val in props.items():
        config.set_property(key, val)
    result = proc.run(config)
    return result


def call_curves_spline(drawable_id, channel, points):
    """Workaround for GimpDoubleArray binding bug — delegate to Script-Fu."""
    pdb = Gimp.get_pdb()
    pts_str = " ".join(str(p) for p in points)
    script = f"(gimp-drawable-curves-spline {drawable_id} {channel} #({pts_str}))"
    proc = pdb.lookup_procedure('plug-in-script-fu-eval')
    config = proc.create_config()
    config.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
    config.set_property('script', script)
    return proc.run(config)


def load_image(path):
    """Load image, return (image, drawable)."""
    result = call_pdb('gimp-file-load',
        **{'run-mode': Gimp.RunMode.NONINTERACTIVE,
           'file': Gio.File.new_for_path(path)})
    image = result.index(1)
    drawable = image.get_selected_drawables()[0]
    return image, drawable


def save_image(image, path):
    """Save image to path."""
    call_pdb('gimp-file-save',
        **{'run-mode': Gimp.RunMode.NONINTERACTIVE,
           'image': image,
           'file': Gio.File.new_for_path(path)})
