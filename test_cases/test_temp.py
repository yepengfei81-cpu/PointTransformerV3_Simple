import torch

# 🔥 关键：加上 weights_only=False
data = torch.load(
    "/root/autodl-tmp/touch_processed_data/Scissors/patches/patch_000001.pth",
    weights_only=False  # 允许加载 numpy 数组
)

print("Keys:", data.keys())
print("category_id:", data.get("category_id", "NOT FOUND"))
print("category_id type:", type(data.get("category_id")))
print("category_id value:", data.get("category_id"))

# 🔥 额外：显示期望值
expected_id = 0  # Scissors 应该是 0
actual_id = data.get("category_id")
print(f"\n期望值: {expected_id} (Scissors)")
print(f"实际值: {actual_id}")
print(f"是否正确: {'✅ 正确' if actual_id == expected_id else '❌ 错误'}")