import onnx
import onnx_graphsurgeon as gs
import numpy as np

model = onnx.load("/home/slon/DCS/egida_mvp/weights/v2_mask_rcnn_r50fpn3x_1333_800_bs1.onnx")
graph  = gs.import_onnx(model)
patched = False

for node in graph.nodes:
    if node.op == "Split" and node.name.endswith("mask_head/Split"):
        print("Patch:", node.name, "→ Identity")
        node.op = "Identity"     # меняем тип узла
        node.attrs.clear()       # убираем axis / split
        patched = True

if not patched:
    print("Nothing patched – Split узел не найден")

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "/home/slon/DCS/egida_mvp/weights/v2_mask_rcnn_r50fpn3x_1333_800_bs1_fixed2.onnx")
