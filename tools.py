import cv2
import torch


class tools:
    @staticmethod
    def view_image(img_path, label, position, sort, out_label, out_position, out_sort):
        new_img = cv2.imread(img_path)

        # 将out_position转换为Tensor类型
        position_tensor = torch.tensor(position)
        if torch.sum(position_tensor) != 0:
            cv2.rectangle(new_img, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 3)
            cv2.putText(new_img, str(sort), (position[0], position[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # 将out_position转换为Tensor类型
        out_position_tensor = torch.tensor(out_position)
        if torch.sum(out_position_tensor) != 0:
            cv2.rectangle(new_img, (out_position[0], out_position[1]), (out_position[2], out_position[3]),   (0, 0, 255), 3)
            cv2.putText(new_img, str(out_sort), (out_position[0], out_position[1] - 3),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('new_img', new_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
