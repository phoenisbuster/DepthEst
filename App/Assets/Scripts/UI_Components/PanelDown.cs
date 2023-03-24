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
    public FeatureOptions FeatureOptionsScript;

    public static Action<bool, bool> ToggleEditMode;
    
    // Start is called before the first frame update
    void Start()
    {
        SendPictureBtn.gameObject.SetActive(false);
    }

    private void OnEnable() 
    {
        FeatureOptions.onFeatureChange += onFeatureChange;
        PanelDown.ToggleEditMode += ShowEditMode;
    }

    private void OnDisable() 
    {
        FeatureOptions.onFeatureChange -= onFeatureChange;
        PanelDown.ToggleEditMode -= ShowEditMode;
    }

    private void onFeatureChange(FeatureList oldType, FeatureList newType)
    {
        switch(newType)
        {
            case FeatureList.BuiltInPhoto:
                BuiltInCameraFunctions.getInstance().clickAccessCamera();
                break;
            
            case FeatureList.RealtimeRender:
                BuiltInCameraFunctions.getInstance().clickAccessCamera();
                break;

            case FeatureList.NativePhoto:
                ShowEditMode(true, false);
                BuiltInCameraFunctions.getInstance().StopWebCamTex();
                NativeFunctions.getInstance().clickCapture();
                break;

            case FeatureList.NativeRecord:
                ShowEditMode(true, false);
                BuiltInCameraFunctions.getInstance().StopWebCamTex();
                NativeFunctions.getInstance().clickRecord();
                break;
        } 
    }

    private void ShowEditMode(bool isShow = false, bool isForceShow = false)
    {
        CapturePictureBtn.gameObject.SetActive(!isShow);
        AccessGalleryBtn.gameObject.SetActive(!isShow);
        ChangeCameraBtn.gameObject.SetActive(!isShow);
        FeatureOptionsScript.gameObject.SetActive(!isShow);

        if(isForceShow)
        {
            SendPictureBtn.gameObject.SetActive(isShow);
        }    
    }

    public void onClikcCapture()
    {
        switch(FeatureOptionsScript.currentFeatureType)
        {
            case FeatureList.BuiltInPhoto:
                ShowEditMode(true, false);
                BuiltInCameraFunctions.getInstance().clickCapture();
                break;
            
            case FeatureList.RealtimeRender:
                ShowEditMode(true, true);
                break;

            case FeatureList.NativePhoto:
                ShowEditMode(true, false);
                NativeFunctions.getInstance().clickCapture();
                break;

            case FeatureList.NativeRecord:
                ShowEditMode(true, false);
                NativeFunctions.getInstance().clickRecord();
                break;
        }  
    }

    public void onClikcChangeCamera()
    {
        if(FeatureOptionsScript.currentFeatureType == FeatureList.BuiltInPhoto ||
            FeatureOptionsScript.currentFeatureType == FeatureList.RealtimeRender)
        {
            BuiltInCameraFunctions.getInstance().changeCameraIndex();
        }    
    }

    public void onClikcAccessGallery()
    {
        BuiltInCameraFunctions.getInstance().StopWebCamTex();
        NativeFunctions.getInstance().GetImageFromGallery();

        ShowEditMode(true, false);
    }

    public void setSendPictureBtn(bool isOn)
    {
        SendPictureBtn.interactable = isOn;
        SendPictureBtn.gameObject.SetActive(isOn);
    }

    public void onClikcUsePicture()
    {
        switch(FeatureOptionsScript.currentFeatureType)
        {
            case FeatureList.BuiltInPhoto:
                ShowEditMode(false, true);
                BuiltInCameraFunctions.getInstance().clickAccessCamera();
                //WSConnection.getInstance().setTargetPos();
                break;
            
            case FeatureList.RealtimeRender:
                ShowEditMode(false, true);
                break;

            case FeatureList.NativePhoto:
                ShowEditMode(false, true);
                BuiltInCameraFunctions.getInstance().ImageScript.hideTexture();
                //WSConnection.getInstance().setTargetPos();
                break;

            case FeatureList.NativeRecord:
                ShowEditMode(false, true);
                BuiltInCameraFunctions.getInstance().ImageScript.hideTexture();
                //WSConnection.getInstance().setTargetPos();
                break;
        }
    }
}
