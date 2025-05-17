def apply_controlnet_to_conditioning(positive, negative, control_net, image, strength,
                                     start_percent=0.0, end_percent=1.0, vae=None, extra_concat=None):
    """
    Apply controlnet to raw conditioning arrays.
    Returns a tuple of (modified_positive, modified_negative).
    This is compatible with the original node.py implementation.
    """
    if strength == 0 or control_net is None:
        return (positive, negative)

    if extra_concat is None:
        extra_concat = []

    control_hint = image.movedim(-1, 1)

    # Process positive conditioning
    modified_pos = []
    for t in positive:
        d = t[1].copy()
        prev_cnet = d.get('control', None)

        temp_cnet = control_net.set_cond_hint(control_hint, strength,
                                              (start_percent, end_percent),
                                              vae=vae, extra_concat=extra_concat)

        if prev_cnet is not None:
            temp_cnet.set_previous_controlnet(prev_cnet)

        d['control'] = temp_cnet
        d['control_apply_to_uncond'] = False
        modified_pos.append([t[0], d])

    # Process negative conditioning (same logic)
    modified_neg = []
    for t in negative:
        d = t[1].copy()
        prev_cnet = d.get('control', None)

        temp_cnet = control_net.set_cond_hint(control_hint, strength,
                                              (start_percent, end_percent),
                                              vae=vae, extra_concat=extra_concat)

        if prev_cnet is not None:
            temp_cnet.set_previous_controlnet(prev_cnet)

        d['control'] = temp_cnet
        d['control_apply_to_uncond'] = False
        modified_neg.append([t[0], d])

    return (modified_pos, modified_neg)