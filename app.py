import streamlit as st
            probs = results[0].probs
            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            pred_class = list(classes.keys())[cls_id]
            label = f"{classes[pred_class]} ({conf:.2f})"
            cv2.putText(img, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Jalankan kamera dengan WebRTC
    webrtc_streamer(
        key="emotion-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: EmotionProcessor(model, confidence_threshold, iou_threshold),
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com"]},
                {"urls": ["stun:stun.nextcloud.com:3478"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:stun.voipbuster.com:3478"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Izinkan akses kamera di browser Anda. "
            "Jika tidak muncul gambar, pastikan menggunakan HTTPS atau localhost, "

            "dan cek permission kamera di browser.")
