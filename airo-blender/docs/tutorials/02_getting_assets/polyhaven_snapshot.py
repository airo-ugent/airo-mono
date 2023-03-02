import json

import airo_blender as ab

all_assets = ab.available_assets()
polyhaven_assets = [asset for asset in all_assets if asset["library"] == "Poly Haven"]

asset_snapshot = {"assets": polyhaven_assets}

with open("asset_snapshot.json", "w") as file:
    json.dump(asset_snapshot, file, indent=4)
