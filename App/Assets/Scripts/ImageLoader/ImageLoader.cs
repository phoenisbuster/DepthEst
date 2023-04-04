using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public enum CameraState
{
    Disable,
    FrontCam,
    BackCam,
    Display
}

public enum DeviceType
{
    Editor,
    IOS,
    Android
}

public class ImageLoader : MonoBehaviour
{
    public RawImage rawimage;
    public Image squareHover;
    public float ImageWidth;
    public float ImageHeight;
    public int TextureWidth;
    public int TextureHeight;
    public CameraState currentState = CameraState.Disable;
    public DeviceType currentDevice = DeviceType.Editor;
    public static Action<CameraState> changeStateSignal;

    // Start is called before the first frame update
    void Awake() 
    {
        #if UNITY_IOS
            TextureWidth = 360*2;
            TextureHeight = 480*2;
            currentDevice = DeviceType.IOS;
            changeCameraState(CameraState.BackCam);
            changeImageSize();
            Debug.Log("Device IOS");
        #endif

        #if UNITY_EDITOR
            TextureWidth = 640*2;
            TextureHeight = 480*2;
            currentDevice = DeviceType.Editor;
            changeCameraState(CameraState.FrontCam);
            RotateImage(false, 0);
            Debug.Log("Device Editor");
        #endif

        #if UNITY_ANDROID
            TextureWidth = 480*2;
            TextureHeight = 640*2;
            currentDevice = DeviceType.Android;
            changeCameraState(CameraState.BackCam);
            changeImageSize();
            Debug.Log("Device Android");
        #endif
    }

    void Start()
    {
        changeCameraState(CameraState.Disable);
    }

    private void OnEnable() 
    {
        ImageLoader.changeStateSignal += changeCameraState;
    }

    private void OnDisable() 
    {
        ImageLoader.changeStateSignal -= changeCameraState;
    }

    public void changeCameraState(CameraState newState)
    {
        currentState = newState;
        switch(currentState)
        {
            case CameraState.Disable:
                hideTexture();
                break;
            case CameraState.FrontCam:
                showTexture();
                if(currentDevice == DeviceType.Editor)
                {
                    ChangeScaleXminus1();
                }
                else if(currentDevice == DeviceType.IOS)
                {
                    ChangeScaleX1();
                    ChangeScaleY1();
                }
                else
                {
                    ChangeScaleXminus1();
                    ChangeScaleY1();
                }
                break;
            case CameraState.BackCam:
                showTexture();
                if(currentDevice == DeviceType.Editor)
                {
                    return;
                }
                else if(currentDevice == DeviceType.IOS)
                {
                    ChangeScaleX1();
                    ChangeScaleYminus1();
                }
                else
                {
                    ChangeScaleX1();
                    ChangeScaleY1();
                }
                break;
            case CameraState.Display:
                if(currentDevice != DeviceType.Editor)
                {
                    ChangeScaleX1();
                    ChangeScaleY1();
                }
                break;  
        }
    }

    public Vector2 getHoverPosition()
    {
        Debug.Log(squareHover.transform.localPosition);
        return squareHover.transform.localPosition;
    }

    public Vector2 getHoverPositionInPixel()
    {
        Debug.Log(squareHover.transform.GetComponent<RectTransform>().anchoredPosition);
        return squareHover.transform.GetComponent<RectTransform>().anchoredPosition;
    }

    public void changeImageSize()
    {
        if(rawimage.transform.eulerAngles.z == 0)
        {
            ImageWidth = TextureWidth;
            ImageHeight = TextureHeight;
        }
        else
        {
            ImageWidth = TextureHeight;
            ImageHeight = TextureWidth;
        }
        rawimage.rectTransform.sizeDelta = new Vector2(ImageWidth, ImageHeight);
    }

    public void changeColor(Color value)
    {
        rawimage.color = value;
    }

    public void setTexture(WebCamTexture value)
    {
        rawimage.texture = value;
    }

    public void removeTexture()
    {
        rawimage.texture = null;
    }

    public void removeAllTexture()
    {
        rawimage.texture = null;
        rawimage.material.mainTexture = null;
    }

    public void showTexture()
    {
        changeColor(Color.white);
    }

    public void hideTexture()
    {
        changeColor(Color.black);
    }

    public void RotateImage(bool toggle, float angle = 0)
    {
        Debug.Log(rawimage.transform.eulerAngles.z);
        var newZ = angle;
        if(toggle)
        {
            newZ = 0;
            if(rawimage.transform.eulerAngles.z == newZ)
            {
                newZ = -90;
            }
        }
        rawimage.transform.eulerAngles = new Vector3(0, 0, newZ);
        changeImageSize();
    }

    public void ChangeScaleX1()
    {
        var y = rawimage.rectTransform.localScale.y;
        rawimage.rectTransform.localScale = new Vector3(1,y,1);
    }

    public void ChangeScaleXminus1()
    {
        var y = rawimage.rectTransform.localScale.y;
        rawimage.rectTransform.localScale = new Vector3(-1,y,1);
    }

    public void ChangeScaleY1()
    {
        var x = rawimage.rectTransform.localScale.x;
        rawimage.rectTransform.localScale = new Vector3(x,1,1);
    }

    public void ChangeScaleYminus1()
    {
        var x = rawimage.rectTransform.localScale.x;
        rawimage.rectTransform.localScale = new Vector3(x,-1,1);
    }

    void Update() 
    {
        if(Input.GetKeyDown(KeyCode.LeftShift))
        {
            Debug.Log("Img Loader check Width: " + rawimage.texture.width);
            Debug.Log("Img Loader check Height: " + rawimage.texture.height);
        }
    }
}
