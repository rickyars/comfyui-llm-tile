import json


def parse_tile_prompts(json_string, grid_width, grid_height):
    """
    Parse JSON tile prompts for the Tiled Image Generator.

    Args:
        json_string: JSON string containing tile prompts
        grid_width: Width of the grid in tiles
        grid_height: Height of the grid in tiles

    Returns:
        List of tile prompts in the correct order
    """
    try:
        data = json.loads(json_string)

        # Validate the JSON structure
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of tile objects")

        expected_tiles = grid_width * grid_height
        if len(data) != expected_tiles:
            print(f"Warning: Expected {expected_tiles} tile prompts, got {len(data)}. Proceeding with available data.")

        # Extract the prompts in the correct order
        tile_prompts = []
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                if idx < len(data):
                    tile_info = data[idx]

                    # Check for required fields
                    if "prompt" not in tile_info:
                        raise ValueError(f"Tile at index {idx} is missing required 'prompt' field")

                    # If position is missing or incorrect, print a warning but proceed
                    if "position" not in tile_info:
                        print(
                            f"Warning: Tile at index {idx} is missing 'position' field. Using grid position ({x + 1},{y + 1}).")
                    else:
                        pos = tile_info["position"]
                        expected_x = x + 1
                        expected_y = y + 1
                        if pos.get("x") != expected_x or pos.get("y") != expected_y:
                            print(
                                f"Warning: Tile at index {idx} has position {pos} but expected ({expected_x},{expected_y}). Using prompt anyway.")

                    tile_prompts.append(tile_info["prompt"])
                else:
                    # If we're missing data, use a default prompt
                    default_prompt = f"Generate content for tile at position ({x + 1},{y + 1})"
                    print(f"Warning: Missing data for tile at index {idx}. Using default prompt.")
                    tile_prompts.append(default_prompt)

        return tile_prompts

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error parsing JSON: {str(e)}")