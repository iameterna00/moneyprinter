from moviepy.editor import TextClip, CompositeVideoClip

def create_pop_text_clip(txt, duration=5, font="../../fonts/luck.ttf", fontsize=50,
                     color="white", stroke_color="#0f0f0f", stroke_width=3,
                     shadow_color="black", shadow_offset=(5, 5), pop_duration=0.1, shadow_opacity=0.1):
    """
    Creates a simple text clip with shadow (no animation)
    """

    main_clip = TextClip(
        txt,
        font=font,
        fontsize=fontsize,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        size=(700, None),
        method="caption",
        align="center"
    ).set_duration(duration).set_position("center")

    shadow_clip = TextClip(
        txt,
        font=font,
        fontsize=fontsize,
        color=shadow_color,
        stroke_color=shadow_color,
        stroke_width=stroke_width,
        size=(700, None),
        method="caption",
        align="center"
    ).set_duration(duration).set_position(
        ("center", shadow_offset[1])
    ).fl_image(lambda pic: (pic * shadow_opacity).astype('uint8'))

    # Combine shadow + main (no animation)
    text_clip = CompositeVideoClip([shadow_clip, main_clip])
    return text_clip.set_duration(duration)


if __name__ == "__main__":
    # Create simple text with shadow
    text_clip = create_text_clip(
        "Simple Text", 
        duration=3,
        fontsize=70,
        color="#FF5252",
        stroke_color="#D32F2F",
        stroke_width=3,
        shadow_color="#000000",
        shadow_offset=(5, 5),
        shadow_opacity=0.2
    )
    
    # Preview the clip
    text_clip.preview(fps=24)
    
    # Or save to file
    # text_clip.write_videofile("simple_text.mp4", fps=24)