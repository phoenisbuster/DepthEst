using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DG.Tweening;

public class FeaturesInstance : MonoBehaviour
{
    public FeatureList featureType;
    public Button BtnComp;
    public TextMeshProUGUI LabelComp;
    public float CenterPosition;
    public static Action<FeatureList> onClickSignal;
    
    public void onCLickBtn()
    {
        FeaturesInstance.onClickSignal.Invoke(featureType);
    }

    public void changeColor(Color value)
    {
        LabelComp.color = value;
    }

    public void changeBtnAvaiability(bool value = true)
    {
        BtnComp.interactable = value;
    }

    public void changeScale(float value = 1, bool isTween = true)
    {
        if(isTween)
        {
            transform.DOScale(new Vector3(value, value, 1), 0.5f).SetLink(gameObject);
        }
        else
        {
            transform.localScale = new Vector3(value, value, 1);
        }
    }

    public void setFeatureAttr(Color color, bool isClickable = true)
    {
        changeColor(color);
        changeBtnAvaiability(isClickable);
    }
}
