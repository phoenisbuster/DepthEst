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
    
    // Start is called before the first frame update
    void Start()
    {
        SendPictureBtn.gameObject.SetActive(false);
    }

    private void OnEnable() 
    {
        FeatureOptions.onFeatureChange += onFeatureChange;
    }

    private void OnDisable() 
    {
        FeatureOptions.onFeatureChange -= onFeatureChange;
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
                BuiltInCameraFunctions.getInstance().StopWebCamTex();
                break;

            case FeatureList.NativeRecord:
                BuiltInCameraFunctions.getInstance().StopWebCamTex();
                break;
        } 
    }

    private void ShowEditMode(bool isShow = false)
    {
        CapturePictureBtn.gameObject.SetActive(!isShow);
        AccessGalleryBtn.gameObject.SetActive(!isShow);
        ChangeCameraBtn.gameObject.SetActive(!isShow);
        FeatureOptionsScript.gameObject.SetActive(!isShow);

        SendPictureBtn.gameObject.SetActive(isShow);
    }

    public void onClikcCapture()
    {
        switch(FeatureOptionsScript.currentFeatureType)
        {
            case FeatureList.BuiltInPhoto:
                BuiltInCameraFunctions.getInstance().clickCapture();
                ShowEditMode(true);
                break;
            
            case FeatureList.RealtimeRender:
                ShowEditMode(true);
                break;

            case FeatureList.NativePhoto:
                NativeFunctions.getInstance().clickCapture();
                ShowEditMode(true);
                break;

            case FeatureList.NativeRecord:
                NativeFunctions.getInstance().clickRecord();
                ShowEditMode(true);
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

        ShowEditMode(true);
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
                Debug.Log("CHECK");
                BuiltInCameraFunctions.getInstance().clickAccessCamera();
                ShowEditMode(false);
                break;
            
            case FeatureList.RealtimeRender:
                ShowEditMode(false);
                break;

            case FeatureList.NativePhoto:
                BuiltInCameraFunctions.getInstance().ImageScript.hideTexture();
                ShowEditMode(false);
                break;

            case FeatureList.NativeRecord:
                BuiltInCameraFunctions.getInstance().ImageScript.hideTexture();
                ShowEditMode(false);
                break;
        }
    }
}
