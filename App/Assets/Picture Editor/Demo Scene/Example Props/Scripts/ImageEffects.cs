using UnityEngine;
using System.IO;


public class ImageEffects : MonoBehaviour
{

    public static ImageEffects instance;

    public MeshRenderer testObject;

    public Texture2D testCutout;
    
    public ImageEffect imageOne;

    public bool compress;
    public int compressTo;
    public bool toImage;
    public bool toCircle;
    public bool toSquare;
    public bool toEllipse;
    public bool toCutout;
    public string fileName;
    public bool save;

    void Update()
    {
        if (toImage)
        {
            toImage = false;
            imageOne.sourceMesh.sharedMaterial.mainTexture = imageOne.sourceTex;
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (compress)
        {
            compress = false;
            imageOne.CondenceTexture(compressTo);
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (toCircle)
        {
            toCircle = false;
            imageOne.ToCircle();
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (toSquare)
        {
            toSquare = false;
            imageOne.ToSquare();
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (toEllipse)
        {
            toEllipse = false;
            imageOne.ToEllipse();
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (toCutout)
        {
            toCutout = false;
            imageOne.CustomCutout(testCutout);
            testObject.transform.localScale = new Vector3((float)testObject.sharedMaterial.mainTexture.width / (float)testObject.sharedMaterial.mainTexture.height, 1, 1);
        }
        if (save)
        {
            save = false;
            imageOne.saveFile(fileName, testObject.sharedMaterial.mainTexture as Texture2D);
        }

    }
    
    void Awake()
    {
        MakeSingleton();
    }

    void MakeSingleton()
    {
        if (instance != null)
        {
            Destroy(gameObject);
        }
        else
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
    }
    
}
