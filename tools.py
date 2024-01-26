import cv2

class tools:
    def view_image( img_path, label, position, sort, out_label, out_position, out_sort):
        new_img = cv2.imread(img_path)
        cv2.rectangle(new_img, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 3)
        cv2.putText(new_img, str(sort), (position[0], position[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        if out_label > 0.5:
            cv2.rectangle(new_img, (out_position[0], out_position[1]), (out_position[2], out_position[3]),
                          (0, 0, 255), 3)
            cv2.putText(new_img, str(out_sort), (out_position[0], out_position[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('new_img', new_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
