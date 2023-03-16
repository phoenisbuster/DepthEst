using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DG.Tweening;

public class SettingPanel : MonoBehaviour
{
    public ImageLoader ImageScript;
    public TextMeshProUGUI Title_Address;
    public TextMeshProUGUI Title_Resolution;
    public TMP_InputField AddressInput;
    public Button SaveAddressBtn;
    public TMP_Dropdown ResolutionOption;
    public TextMeshProUGUI DebugLog;
    public Canvas MainCanvas;
    public float defaultWidth = 720;
    public float defaultHeight = 1280;

    private string currentAddress = "";
    private string savedAddress = "";
    private float currentWidth = 720;
    private float currentHeight = 1280;
    private int currentOption = 0;

    private void Awake() 
    {
        defaultWidth = MainCanvas.GetComponent<RectTransform>().rect.width;
        defaultHeight = MainCanvas.GetComponent<RectTransform>().rect.height;
        ResolutionOption.options[0].text = "Canvas (" + defaultWidth + "x" + defaultHeight + ")";

        currentAddress = UserData.GetServerAddress();
        currentOption = UserData.GetResolutionOption();
        if(currentAddress != "")
        {
            AddressInput.text = currentAddress;
            savedAddress = currentAddress;
            WSConnection.getInstance().changeAddress(savedAddress);
        }
        ResolutionOption.value = currentOption;
    }

    private void Start() 
    {
        OnResolutionChange();
    }

    public void OnResolutionChange()
    {
        if(ResolutionOption.value == 0)
        {
            currentWidth = defaultWidth;
            currentHeight = defaultHeight;
        }
        else
        {
            currentWidth = Int32.Parse(ResolutionOption.options[ResolutionOption.value].text.Split('x')[0]);
            currentHeight = Int32.Parse(ResolutionOption.options[ResolutionOption.value].text.Split('x')[1]);
        }
        Debug.Log(currentWidth + " " + currentHeight);
        ImageScript.changeImageSize(currentWidth, currentHeight);

        UserData.SaveResolutionOption(ResolutionOption.value);
    }

    public void OnInputAddressFinish()
    {
        currentAddress = AddressInput.text;
    }

    public void OnClickSaveAddress()
    {
        savedAddress = currentAddress;
        UserData.SaveServerAddress(savedAddress);
        WSConnection.getInstance().changeAddress(savedAddress, true);
    }
}