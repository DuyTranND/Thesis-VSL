import streamlit as st
import requests
import os

st.set_page_config(layout="centered")

st.title("️📹 Nhận diện Ngôn ngữ Ký hiệu")

# Video uploader
uploaded_file = st.file_uploader("Chọn một video (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Hiển thị video đã tải lên
    st.video(uploaded_file)

    # Nút dự đoán
    if st.button("Bắt đầu nhận diện 🔎"):
        with st.spinner("⏳ Đang xử lý, vui lòng chờ..."):
            try:
                # Địa chỉ endpoint của FastAPI backend
                # Đảm bảo backend của bạn đang chạy ở địa chỉ này
                api_url = "http://127.0.0.1:8000/predict"

                # Gửi video đến backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(api_url, files=files, timeout=60) # Thêm timeout để tránh chờ quá lâu

                # Xử lý kết quả trả về
                if response.status_code == 200:
                    predictions = response.json()

                    if predictions:
                        # Lấy dự đoán có độ tin cậy cao nhất
                        top_prediction = predictions[0]
                        top_word = top_prediction['word'].replace('_', ' ').title()
                        top_confidence = top_prediction['confidence']

                        # Hiển thị kết quả chính
                        st.success(f"🏆 Dự đoán chính: **{top_word}**")
                        st.metric(label="Độ tin cậy", value=f"{top_confidence:.2%}")

                        # Hiển thị các dự đoán khác nếu có
                        if len(predictions) > 1:
                            st.markdown("---")
                            st.subheader("Các khả năng khác:")
                            
                            # Bỏ qua dự đoán đầu tiên đã hiển thị
                            for pred in predictions[1:]:
                                word = pred['word'].replace('_', ' ').title()
                                confidence = pred['confidence']
                                st.write(f"{pred['rank']}. **{word}** - *({confidence:.2%})*")
                    else:
                        st.warning("Không nhận diện được từ nào trong video.")

                else:
                    # Hiển thị lỗi từ server một cách chi tiết hơn
                    st.error(f"Lỗi từ server: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi kết nối đến server: {e}")
                st.info("Hãy đảm bảo rằng bạn đã khởi chạy server backend và địa chỉ API là chính xác.")
            except Exception as e:
                st.error(f"Đã có lỗi xảy ra: {e}")