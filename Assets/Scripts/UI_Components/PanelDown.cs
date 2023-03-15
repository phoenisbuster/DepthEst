using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DG.Tweening;

public class PanelDown : MonoBehaviour
{
    public Button CapturePictureBtn;
    public Button AccessGalleryBtn;
    public Button ChangeCameraBtn;
    public Button SendPictureBtn;
    public ScrollRect FeatureOptions;
    
    // Start is called before the first frame update
    void Start()
    {
        SendPictureBtn.interactable = false;
        SendPictureBtn.gameObject.SetActive(false);
    }

    public void setSendPictureBtn(bool isOn)
    {
        SendPictureBtn.interactable = isOn;
        SendPictureBtn.gameObject.SetActive(isOn);
    }

    public void onClikcCapture()
    {
        BuiltInCameraFunctions.getInstance().clickCapture();
    }

    public void onClikcChangeCamera()
    {
        BuiltInCameraFunctions.getInstance().changeCameraIndex();
    }
}
