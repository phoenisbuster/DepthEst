using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DG.Tweening;

public enum FeatureList
{
    BuiltInPhoto,
    RealtimeRender,
    NativePhoto,
    NativeRecord
}

public class FeatureOptions : MonoBehaviour
{
    public ScrollRect ScrollRectComp;
    public GameObject ContentComp;
    public float ScaleWhenClick = 1.5f;
    public FeatureList currentFeatureType = FeatureList.BuiltInPhoto;
    private Dictionary<FeatureList, FeaturesInstance> featureList = new Dictionary<FeatureList, FeaturesInstance>();

    void Awake() 
    {
        Debug.Log(ContentComp.transform.position);
        Debug.Log(ContentComp.transform.localPosition);
        for(int i = 0; i < ContentComp.transform.childCount; i++)
        {
            var featureType = ContentComp.transform.GetChild(i).GetComponent<FeaturesInstance>().featureType;
            featureList.Add(featureType, ContentComp.transform.GetChild(i).GetComponent<FeaturesInstance>());
        }
    }
    
    // Start is called before the first frame update
    void Start()
    {
        initDefaultFeature();
    }

    private void OnEnable() 
    {
        FeaturesInstance.onClickSignal += setFeature;
    }

    private void OnDisable()
    {
        FeaturesInstance.onClickSignal -= setFeature;
    }

    private void initDefaultFeature()
    {
        featureList[currentFeatureType].setFeatureAttr(Color.yellow, false);
        featureList[currentFeatureType].changeScale(ScaleWhenClick, false);
    }

    private void setFeature(FeatureList type)
    {
        if(currentFeatureType == type)
        {
            return;
        }

        featureList[currentFeatureType].setFeatureAttr(Color.white, true);
        featureList[currentFeatureType].changeScale(1, true);

        currentFeatureType = type;
        playAnim(ContentComp, featureList[currentFeatureType].CenterPosition);
    }

    private void playAnim(GameObject target, float value)
    {
        featureList[currentFeatureType].changeScale(ScaleWhenClick, true);
        featureList[currentFeatureType].setFeatureAttr(Color.yellow, false);
        target.transform.DOLocalMoveX(value, 0.5f).SetEase(Ease.InOutSine).SetLink(ContentComp);
    }
}
