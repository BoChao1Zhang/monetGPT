from gimp3_compat import call_pdb, call_curves_spline
import numpy as np


def adjust_tint(image, drawable, intensity):
    """
    Adjust the image tint based on intensity.
    intensity: float in [-1.0, 1.0]
        -1.0 = More green
         0.0 = No change
         1.0 = More magenta
    """
    # Ensure intensity is within [-1, 1]
    intensity = max(-1.0, min(1.0, intensity))
    # Scaling factor to translate intensity to color balance value
    factor = 30.0
    # Other axes remain unchanged
    cyan_red = 0.0
    magenta_green = intensity * factor  # +ve: more magenta, -ve: more green
    yellow_blue = 0.0
    transfer_mode = 1  # Affect mid-tones
    preserve_lum = True
    # Apply the color balance to affect tint
    call_pdb('gimp-drawable-color-balance',
        drawable=drawable, **{'transfer-mode': transfer_mode,
        'preserve-lum': preserve_lum, 'cyan-red': cyan_red,
        'magenta-green': magenta_green, 'yellow-blue': yellow_blue})


def adjust_temperature(image, drawable, intensity):
    """
    Adjust the image temperature based on intensity.
    intensity: float in [-1.0, 1.0]
        -1.0 = Very cool (more cyan/blue)
         0.0 = No change
         1.0 = Very warm (more red/yellow)
    """
    intensity = intensity * 2.5
    # Chosen scaling factor to translate intensity to color balance values
    red_factor = 15.0
    yellow_factor = 30.0
    # Positive intensity: warm (more red, more yellow)
    # Negative intensity: cool (more cyan, more blue)
    cyan_red = intensity * red_factor  # +ve: more red, -ve: more cyan
    magenta_green = 0.0  # No change
    yellow_blue = intensity * (
        -yellow_factor
    )  # +ve: more yellow (because negative sign), -ve: more blue
    preserve_lum = True
    # Apply color balance to midtones, highlights, shadows
    for transfer_mode in [1, 2, 0]:
        call_pdb('gimp-drawable-color-balance',
            drawable=drawable, **{'transfer-mode': transfer_mode,
            'preserve-lum': preserve_lum, 'cyan-red': cyan_red,
            'magenta-green': magenta_green, 'yellow-blue': yellow_blue})


def adjust_clarity(image, drawable, intensity):
    """
    Adjust the clarity of the image.
    intensity: float in [-1.0, 1.0]
      -1.0 = Decrease clarity (soften)
       0.0 = No change
       1.0 = Increase clarity (add local contrast)

    NOTE: plug-in-unsharp-mask and plug-in-gauss-rle are not available in
    GIMP 3.0. This function needs GEGL equivalents to be implemented.
    """
    # Not available in GIMP 3.0 — needs GEGL equivalents
    pass


def adjust_light(
    drawable, shadows, highlights, whites, compress=75, shadow_correct=50, hl_correct=50
):
    radius = 300
    shadows = shadows * 100
    highlights = highlights * 100
    whitepoint_factor = 2.7 if whites > 0 else 1.5
    whites = int(whites * 10 * whitepoint_factor)
    whitepoint_sign = 1 if whites >= 0 else -1
    whites = abs(whites)
    white_0 = min(whites, 10) * whitepoint_sign
    call_pdb('gimp-drawable-shadows-highlights',
        drawable=drawable, shadows=float(shadows), highlights=float(highlights),
        whitepoint=float(white_0), radius=float(radius), compress=float(compress),
        **{'shadows-ccorrect': float(shadow_correct),
           'highlights-ccorrect': float(hl_correct)})
    whites = whites - 10
    shadows = 0
    highlights = 0
    while whites > 0:
        residue = min(whites, 10) * whitepoint_sign
        call_pdb('gimp-drawable-shadows-highlights',
            drawable=drawable, shadows=float(shadows), highlights=float(highlights),
            whitepoint=float(residue), radius=float(radius), compress=float(compress),
            **{'shadows-ccorrect': float(shadow_correct),
               'highlights-ccorrect': float(hl_correct)})
        whites = whites - 10


def adjust_contrast(image, drawable, intensity):
    # - range (-0.2,0.2) contrast
    print("intensssity", intensity)
    call_pdb('gimp-drawable-brightness-contrast',
        drawable=drawable, brightness=0.0, contrast=intensity / 5)


def adjust_saturation(image, drawable, intensity):
    # range (-100,80) saturation
    call_pdb('gimp-drawable-hue-saturation',
        drawable=drawable, **{'hue-range': 0, 'hue-offset': 0.0,
        'lightness': 0.0, 'saturation': float(int(intensity * 100)),
        'overlap': 0.0})


def adjust_sharpen(image, drawable, intensity):
    """
    NOTE: plug-in-sharpen and plug-in-gauss-rle are not available in
    GIMP 3.0. This function needs GEGL equivalents to be implemented.
    """
    # Not available in GIMP 3.0 — needs GEGL equivalents
    pass


def adjust_shadows_highlights(
    image,
    drawable,
    adjust_type,
    intensity,
    compress=75,
    shadow_correct=50,
    hl_correct=50,
):
    """
    Adjust shadows, highlights, or whites of the given drawable.

    Parameters:
        drawable: GIMP drawable object.
        adjust_type: str, one of "shadows", "highlights", or "whitepoint".
        intensity: float, value scaled between -1 and +1 for adjustment intensity.
    Returns:
        None
    """
    # Ensure intensity is scaled between -1 and +1
    if not (-1.0 <= intensity <= 1.0):
        raise ValueError("Intensity must be between -1 and +1.")
    # Map intensity to the appropriate ranges for each parameter
    if adjust_type == "shadows":
        shadows = intensity * 100  # Scale to -100 to +100
        highlights = 0
        whitepoint = 0
    elif adjust_type == "highlights":
        shadows = 0
        highlights = intensity * 100  # Scale to -100 to +100
        whitepoint = 0
    elif adjust_type == "whites":
        shadows = 0
        highlights = 0
        # INVERTIBLE
        whitepoint_factor = 2.0
        whitepoint = int(intensity * 10 * whitepoint_factor)  # Scale to -10 to +10
    else:
        raise ValueError(
            "Invalid adjust_type. Must be 'shadows', 'highlights', or 'whites'."
        )
    radius = 300
    if whitepoint >= 0:
        # Call the internal GIMP function
        while whitepoint >= 0:
            whitepoint_val = min(whitepoint, 10)
            call_pdb('gimp-drawable-shadows-highlights',
                drawable=drawable, shadows=float(shadows),
                highlights=float(highlights),
                whitepoint=float(min(whitepoint, 10)),
                radius=float(radius), compress=float(compress),
                **{'shadows-ccorrect': float(shadow_correct),
                   'highlights-ccorrect': float(hl_correct)})
            whitepoint = whitepoint - 10
    else:
        while whitepoint < 0:
            whitepoint_val = max(whitepoint, -10)
            call_pdb('gimp-drawable-shadows-highlights',
                drawable=drawable, shadows=float(shadows),
                highlights=float(highlights),
                whitepoint=float(whitepoint_val),
                radius=float(radius), compress=float(compress),
                **{'shadows-ccorrect': float(shadow_correct),
                   'highlights-ccorrect': float(hl_correct)})
            whitepoint = whitepoint + 10


def adjust_blacks(image, drawable, intensity):
    """
    Adjust blacks using a spline curve.
    intensity: float between -1 and +1
        -1 -> deeper blacks
         0 -> no change (straight line)
        +1 -> lifted blacks
    """
    y = 30
    # INVERTIBLE
    if intensity < 0:
        x = y + ((abs(intensity) + 1) ** 4) * 6
    else:
        x = y + ((abs(intensity) + 1) ** 4) * 6
        x, y = y, x
    controls = [0, 0, x, y, 255, 255]
    # Normalize controls for gimp_drawable_curves_spline (0..1 range)
    normalized_controls = [c / 255.0 for c in controls]
    # channel=HISTOGRAM-VALUE via Script-Fu workaround
    call_curves_spline(drawable.get_id(), 'HISTOGRAM-VALUE', normalized_controls)
