import cv2


def draw_text(img,
              text,
              pos,
              fontFace,
              fontScale,
              thickness,
              lineType,
              color_txt=(0, 0, 0),
              color_bg=None,
              alignH="l",  # l: left, c: center, r: right
              alignV="b",  # t: top, c: center, b: bottom
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    text_w, text_h = text_size
    if alignV == "b":
        # txt_pos_y = round((y + fontScale - 1))
        txt_pos_y = y
    elif alignV == "t":
        # txt_pos_y = round((y + fontScale - 1)) + text_h
        txt_pos_y = y + text_h
    elif alignV == "c":
        txt_pos_y = y + text_h // 2
    else:
        raise ValueError("alignV must be one of 't', 'b', 'c'")


    if alignH == "l":
        txt_pos_x = x
    elif alignH == "r":
        txt_pos_x = x - text_w
    elif alignH == "c":
        txt_pos_x = x - text_w // 2
    else:
        raise ValueError("alignH must be one of 'l', 'r', 'c'")

    text_position = (txt_pos_x, txt_pos_y)
    padding = 3
    rect_pos_x = txt_pos_x - padding
    rect_pos_y = txt_pos_y + padding
    rect_position = (rect_pos_x, rect_pos_y)
    if color_bg is not None:
        rect_w = text_w + padding
        rect_h = text_h + padding
        cv2.rectangle(img, rect_position, (txt_pos_x + rect_w, txt_pos_y - rect_h), color_bg, -1)

    cv2.putText(img, text, text_position, fontFace=fontFace, fontScale=fontScale, color=color_txt, thickness=thickness, lineType=lineType)
    return text_size
