from src.common import xyxy_to_yolo
def test_xyxy_to_yolo():
 assert xyxy_to_yolo((0,0,100,50),100,50)==(0.5,0.5,1.0,1.0)
