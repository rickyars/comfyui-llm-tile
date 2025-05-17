def combine_guider_conditioning(guider, tile_prompt, clip):
    """
    Combines a tile-specific prompt with an existing guider's conditioning.
    The tile prompt is added before the global conditioning.

    Args:
        guider: A guider instance (CFGGuider or BasicGuider)
        tile_prompt: The tile-specific prompt to add
        clip: CLIP model for tokenizing and encoding

    Returns:
        original_conditioning: The original conditioning to restore later
    """
    # Determine the guider type
    guider_type = guider.__class__.__name__ if hasattr(guider, "__class__") else "Unknown"

    # Save original conditioning to restore later
    original_conditioning = None
    try:
        if guider_type == "CFGGuider":
            if hasattr(guider, 'positive_cond') and hasattr(guider, 'negative_cond'):
                original_conditioning = (guider.positive_cond.copy(), guider.negative_cond.copy())
        elif hasattr(guider, 'conditioning'):
            original_conditioning = guider.conditioning.copy()
    except Exception as e:
        print(f"Warning: Could not save original conditioning: {e}")

    # Update conditioning for this tile - put tile prompt first, followed by guider's original prompt
    try:
        if guider_type == "CFGGuider":
            # For CFGGuider, encode and combine with original conditioning
            pos_tokens = clip.tokenize(tile_prompt)
            tile_cond = clip.encode_from_tokens_scheduled(pos_tokens)

            # Combine with original (tile specific first, then global)
            if hasattr(guider, 'positive_cond'):
                # We need to combine the conditioning, not replace it
                combined_pos_cond = []

                # Add the tile-specific conditioning first
                combined_pos_cond.extend(tile_cond)

                # Then add the original global conditioning
                for t in guider.positive_cond:
                    combined_pos_cond.append(t)

                # Keep the original negative conditioning
                neg_cond = guider.negative_cond if hasattr(guider, 'negative_cond') else None
            else:
                print("Warning: CFGGuider has no positive_cond attribute")
                combined_pos_cond = tile_cond
                neg_cond = None

            # Set the combined conditioning
            if hasattr(guider, 'set_conds'):
                guider.set_conds(combined_pos_cond, neg_cond)
            elif hasattr(guider, 'positive_cond'):
                guider.positive_cond = combined_pos_cond
                if neg_cond and hasattr(guider, 'negative_cond'):
                    guider.negative_cond = neg_cond

        elif hasattr(guider, 'set_conds') or hasattr(guider, 'conditioning'):
            # For BasicGuider, encode and combine with original conditioning
            tokens = clip.tokenize(tile_prompt)
            tile_cond = clip.encode_from_tokens_scheduled(tokens)

            # Combine with original (tile specific first, then global)
            if hasattr(guider, 'conditioning'):
                # We need to combine the conditioning, not replace it
                combined_cond = []

                # Add the tile-specific conditioning first
                combined_cond.extend(tile_cond)

                # Then add the original global conditioning
                for t in guider.conditioning:
                    combined_cond.append(t)
            else:
                print("Warning: Guider has no conditioning attribute")
                combined_cond = tile_cond

            # Set the combined conditioning
            if hasattr(guider, 'set_conds'):
                guider.set_conds(combined_cond)
            elif hasattr(guider, 'conditioning'):
                guider.conditioning = combined_cond

        else:
            print(f"Warning: Unknown guider type {guider_type}, unable to set conditioning")

    except Exception as e:
        print(f"Error updating conditioning: {e}")

    return original_conditioning


def restore_guider_conditioning(guider, original_conditioning):
    """
    Restores a guider's original conditioning.

    Args:
        guider: A guider instance (CFGGuider or BasicGuider)
        original_conditioning: The original conditioning to restore
    """
    if original_conditioning is None:
        return

    guider_type = guider.__class__.__name__ if hasattr(guider, "__class__") else "Unknown"

    try:
        if guider_type == "CFGGuider" and isinstance(original_conditioning, tuple) and len(original_conditioning) == 2:
            if hasattr(guider, 'set_conds'):
                guider.set_conds(original_conditioning[0], original_conditioning[1])
            else:
                guider.positive_cond = original_conditioning[0]
                guider.negative_cond = original_conditioning[1]
        elif hasattr(guider, 'set_conds'):
            guider.set_conds(original_conditioning)
        elif hasattr(guider, 'conditioning'):
            guider.conditioning = original_conditioning
    except Exception as e:
        print(f"Warning: Failed to restore original conditioning: {e}")


def apply_controlnet_to_guider_direct(guider, controlnet, image, strength, vae=None):
    """
    Applies a controlnet directly to a guider's conditioning.

    Args:
        guider: A guider instance (CFGGuider or BasicGuider)
        controlnet: ControlNet model to apply
        image: Image tensor to use as control hint
        strength: Strength of the controlnet effect
        vae: Optional VAE for encoding

    Returns:
        bool: True if successful, False otherwise
    """
    if controlnet is None or strength <= 0:
        return False

    guider_type = guider.__class__.__name__ if hasattr(guider, "__class__") else "Unknown"
    print(f"Applying ControlNet with strength {strength} to {guider_type}")

    try:
        # Let's directly access and modify the guider's conditioning
        if guider_type == "CFGGuider" and hasattr(guider, 'positive_cond'):
            # It's a CFGGuider with direct access to conditioning
            control_hint = image.movedim(-1, 1)

            # Apply to positive conditioning
            modified_pos = []
            for t in guider.positive_cond:
                d = t[1].copy()
                prev_cnet = d.get('control', None)

                temp_cnet = controlnet.set_cond_hint(control_hint, strength,
                                                     (0.0, 1.0), vae=vae)

                if prev_cnet is not None:
                    temp_cnet.set_previous_controlnet(prev_cnet)

                d['control'] = temp_cnet
                d['control_apply_to_uncond'] = False
                modified_pos.append([t[0], d])

            # Apply to negative conditioning if it has it
            modified_neg = []
            if hasattr(guider, 'negative_cond'):
                for t in guider.negative_cond:
                    d = t[1].copy()
                    prev_cnet = d.get('control', None)

                    temp_cnet = controlnet.set_cond_hint(control_hint, strength,
                                                         (0.0, 1.0), vae=vae)

                    if prev_cnet is not None:
                        temp_cnet.set_previous_controlnet(prev_cnet)

                    d['control'] = temp_cnet
                    d['control_apply_to_uncond'] = False
                    modified_neg.append([t[0], d])

            # Set the modified conditioning back to the guider
            guider.positive_cond = modified_pos
            if hasattr(guider, 'negative_cond'):
                guider.negative_cond = modified_neg

            print("ControlNet applied to CFGGuider conditioning")
            return True

        elif hasattr(guider, 'conditioning'):
            # For BasicGuider with a single conditioning
            control_hint = image.movedim(-1, 1)
            modified_cond = []

            for t in guider.conditioning:
                d = t[1].copy()
                prev_cnet = d.get('control', None)

                temp_cnet = controlnet.set_cond_hint(control_hint, strength,
                                                     (0.0, 1.0), vae=vae)

                if prev_cnet is not None:
                    temp_cnet.set_previous_controlnet(prev_cnet)

                d['control'] = temp_cnet
                d['control_apply_to_uncond'] = False
                modified_cond.append([t[0], d])

            guider.conditioning = modified_cond
            print("ControlNet applied to BasicGuider conditioning")
            return True

        else:
            print(f"Warning: Unable to apply ControlNet to guider type {guider_type}")
            return False

    except Exception as e:
        print(f"Error applying ControlNet: {str(e)}")
        return False