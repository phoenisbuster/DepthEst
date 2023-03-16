using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DG.Tweening;

public class PanelUp : MonoBehaviour
{
    public Button RotateImage;
    public Button SettingBtn;

    public static Action OnClickSettingAction;
    
    public void OnClickSetting()
    {
        OnClickSettingAction.Invoke();
    }

    public void OnClickRotate()
    {
        BuiltInCameraFunctions.getInstance().changeImageRotate();
    }
}
