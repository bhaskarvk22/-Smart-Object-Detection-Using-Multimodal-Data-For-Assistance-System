import os
import sys
import time
import argparse
import cv2
import numpy as np
import RPi.GPIO as GPIO

# Global OLED configuration (moved to function to avoid early imports)
OLED_CONFIG = {
    'enabled': False,
    'address': 0x3C,
    'width': 128,
    'height': 64
}

# Buzzer configuration
BUZZER_PIN = 18  # GPIO pin for the buzzer
BUZZER_ENABLED = True  # Set to False to disable buzzer

def setup_hardware():
    """Initialize all hardware components with lazy imports"""
    # Configure GPIO
    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23, GPIO.OUT)  # TRIG
    GPIO.setup(24, GPIO.IN)   # ECHO
    
    # Setup buzzer
    if BUZZER_ENABLED:
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    # Initialize OLED components only if needed
    disp = oled_image = draw = font = None
    
    # Lazy import OLED libraries
    try:
        import board
        import busio
        import adafruit_ssd1306
        from PIL import Image, ImageDraw, ImageFont
        
        OLED_CONFIG['enabled'] = True
        
        i2c = busio.I2C(board.SCL, board.SDA)
        disp = adafruit_ssd1306.SSD1306_I2C(
            OLED_CONFIG['width'], 
            OLED_CONFIG['height'], 
            i2c, 
            addr=OLED_CONFIG['address']
        )
        disp.fill(0)
        disp.show()
        
        oled_image = Image.new('1', (OLED_CONFIG['width'], OLED_CONFIG['height']))
        draw = ImageDraw.Draw(oled_image)
        font = ImageFont.load_default()
    except ImportError as e:
        print(f"OLED display disabled: {str(e)}")
    except Exception as e:
        print(f"OLED initialization failed: {str(e)}")

    return disp, oled_image, draw, font

def activate_buzzer(duration=0.1):
    """Activate the buzzer for a short duration"""
    if not BUZZER_ENABLED:
        return
        
    try:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
    except Exception as e:
        print(f"Buzzer error: {str(e)}")

def get_distance():
    """Measure distance using ultrasonic sensor"""
    GPIO.output(23, False)
    time.sleep(0.05)  # Reduced sleep time
    
    GPIO.output(23, True)
    time.sleep(0.00001)
    GPIO.output(23, False)
    
    timeout = time.time() + 0.1  # 100ms timeout
    while GPIO.input(24) == 0 and time.time() < timeout:
        pulse_start = time.time()
    
    timeout = time.time() + 0.1  # 100ms timeout
    while GPIO.input(24) == 1 and time.time() < timeout:
        pulse_end = time.time()
    
    try:
        distance = (pulse_end - pulse_start) * 17150
        return max(0, min(round(distance, 2), 400))  # Clamp to 0-400cm
    except:
        return 0

def update_oled(disp, image, draw, font, objects, distance):
    """Optimized OLED display update"""
    if not OLED_CONFIG['enabled'] or disp is None:
        return
        
    try:
        # Clear and update in one operation
        draw.rectangle((0, 0, OLED_CONFIG['width'], OLED_CONFIG['height']), fill=0)
        text_lines = [f"Dist: {distance}cm"]
        
        if objects:
            text_lines.append("Objects:")
            text_lines.extend(f"{obj}: {conf}%" for obj, conf in list(objects.items())[:3])
        else:
            text_lines.append("No objects")
        
        # Draw all text at once
        for i, line in enumerate(text_lines[:4]):  # Max 4 lines
            draw.text((0, i*16), line, font=font, fill=255)
        
        disp.image(image)
        disp.show()
    except Exception as e:
        print(f"OLED update error: {str(e)}")

def initialize_camera(source, resolution):
    """Optimized camera initialization with consistent color handling"""
    resW, resH = map(int, resolution.split('x'))
    
    if source.startswith('usb'):
        cap = cv2.VideoCapture(int(source[3:]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        # Set auto white balance and exposure if needed
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        return cap
    
    try:
        from picamera2 import Picamera2
        from libcamera import controls
        
        cap = Picamera2()
        config = cap.create_video_configuration(
            main={"size": (resW, resH), "format": "RGB888"},
            controls={
                "AwbMode": controls.AwbModeEnum.Auto,
                "AeEnable": True,
                "FrameRate": 30.0
            }
        )
        cap.configure(config)
        cap.start()
        return cap
    except ImportError:
        print("Picamera2 not found, using USB camera")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        return cap
    except Exception as e:
        print(f"Picamera2 initialization failed: {str(e)}")
        # Fall back to USB camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        return cap

def enhance_image(frame):
    """Apply basic image enhancement"""
    # Convert to HSV color space for enhancement
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Enhance contrast in the value channel
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def main():
    """Optimized main program execution with improved color handling"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to YOLO model file')
    parser.add_argument('--source', default='picam', help='Camera source (usb0, picam)')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--resolution', default='640x480', help='Camera resolution')
    parser.add_argument('--enhance', action='store_true', help='Enable image enhancement')
    args = parser.parse_args()

    # Load YOLO model first (most critical component)
    try:
        from ultralytics import YOLO
        model = YOLO(args.model, task='detect')
        labels = model.names
    except Exception as e:
        print(f"Failed to load YOLO model: {str(e)}")
        sys.exit(1)

    # Initialize hardware after model is loaded
    disp, oled_image, draw, font = setup_hardware()
    cap = initialize_camera(args.source, args.resolution)

    try:
        while True:
            # Get sensor data
            distance = get_distance()
            
            # Capture frame with proper color handling
            if args.source.startswith('usb'):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB for USB cameras
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                try:
                    frame = cap.capture_array()
                    # Ensure format is RGB (Picamera2 should already be RGB)
                    if frame.shape[2] == 4:  # If RGBA format
                        frame = frame[:, :, :3]  # Drop alpha channel
                except Exception as e:
                    print(f"Frame capture error: {str(e)}")
                    break
            
            # Apply image enhancement if requested
            if args.enhance:
                frame = enhance_image(frame)
            
            # Ensure proper data type for detection
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Run detection
            detected_objects = {}
            try:
                results = model(frame, verbose=False, half=True)  # Enable half precision
                for det in results[0].boxes:
                    if det.conf.item() > args.thresh:
                        label = labels[int(det.cls.item())]
                        conf = int(det.conf.item() * 100)
                        detected_objects[label] = conf
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = det.xyxy.cpu().numpy().squeeze().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf}%", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                # Activate buzzer if objects detected
                if detected_objects and BUZZER_ENABLED:
                    activate_buzzer()
                    
            except Exception as e:
                print(f"Detection error: {str(e)}")
            
            # Update OLED and display
            update_oled(disp, oled_image, draw, font, detected_objects, distance)
            
            # Convert back to BGR for display if needed
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, f"Distance: {distance}cm", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Object Detection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    
    # Cleanup
    if args.source.startswith('usb'):
        cap.release()
    else:
        if hasattr(cap, 'stop'):
            cap.stop()
        if hasattr(cap, 'close'):
            cap.close()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    if OLED_CONFIG['enabled'] and disp is not None:
        disp.fill(0)
        disp.show()

if __name__ == "__main__":
    main()