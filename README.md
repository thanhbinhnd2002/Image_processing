# Image Processing Project: Faster-RCNN with MobileNetV3 Backbone

## Giới thiệu

Dự án này sử dụng mô hình **Faster-RCNN** với **MobileNetV3** làm backbone để xử lý ảnh và thực hiện nhận diện đối tượng. Faster-RCNN là một trong những mô hình mạnh mẽ trong việc phát hiện đối tượng trong ảnh, trong khi MobileNetV3 giúp giảm thiểu kích thước mô hình và tăng tốc độ tính toán, rất phù hợp cho các ứng dụng di động hoặc hệ thống có tài nguyên hạn chế.

## Các tính năng

- **Faster-RCNN**: Mô hình phát hiện đối tượng với tốc độ và độ chính xác cao.
- **MobileNetV3**: Backbone nhẹ giúp tối ưu hóa mô hình cho các ứng dụng thực tế với tài nguyên tính toán hạn chế.
- **Hỗ trợ tùy chỉnh**: Dễ dàng thay đổi các tham số hoặc mở rộng mô hình để sử dụng với các tập dữ liệu khác.

## Yêu cầu

- Python 3.7+
- TensorFlow 2.x hoặc PyTorch (Tùy vào việc bạn sử dụng framework nào)
- Các thư viện phụ thuộc:
  - numpy
  - matplotlib
  - opencv-python
  - tensorflow (hoặc pytorch)
  - PIL

## Cài đặt

1. **Clone repository**:
    ```bash
    git clone https://github.com/thanhbinhnd2002/Image_processing.git
    
    ```

## Cách sử dụng

1. **Huấn luyện mô hình**: 
   Để huấn luyện mô hình của bạn, chạy lệnh sau:
    ```bash
    python train.py --dataset_path <đường dẫn đến dữ liệu> --batch_size 16 --epochs 20
    ```

2. **Dự đoán đối tượng trong ảnh**:
   Sau khi huấn luyện xong, bạn có thể sử dụng mô hình để nhận diện đối tượng trong các ảnh mới:
    ```bash
    python inference_csv.py --image_path <đường dẫn đến ảnh> --model_path <đường dẫn đến mô hình đã huấn luyện>
    ```
    Sau khi huấn 

3. **Dự đoán đối tượng trong video**: 
   Sau khi huấn luyện xong, bạn có thể sử dụng mô hình để nhận diện đối tượng trong các video mới::
    ```bash
    python Inference_Faster_R_CNN_Video.py --dataset_path <đường dẫn đến dữ liệu kiểm tra> --checkpoint_path <đường dẫn đến mô hình đã huấn luyện>
    ```

## Cấu trúc thư mục

```
image-processing-project/
│
├── data/                  # Dữ liệu huấn luyện và kiểm tra
├── trained_models/                # Mô hình đã huấn luyện
├── scripts/               # Các script huấn luyện, dự đoán, đánh giá
│   ├── train.py
│   ├── inference.py
│
├── tensorboard      # lưu quá trình train
└── README.md              # Tài liệu dự án
```

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua email: phamthanhbinh2002nguyenkhuyen@gmail.com.

---

