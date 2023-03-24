using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class NativeFunctions : MonoBehaviour
{
    public RawImage rawimage;
    public ImageLoader ImageScript;

    private static NativeFunctions instance;

    private void Awake() 
    {
        instance = this;
    }

    public static NativeFunctions getInstance()
    {
        return instance;
    }

    void Start() 
    {
        rawimage = ImageScript.rawimage;    
    }

    void OnDestroy() 
    {
        instance = null;
    }

    public void clickCapture()
    {
        TakePicture(2048);
    }

    public void clickRecord()
    {
        RecordVideo();
    }

    public void GetImageFromGallery()
    {
        PickImage(2048);
    }

    public void GetVideoFromGallery()
    {
        PickVideo();
    }

    private bool checkCameraBusy()
    {
        return NativeCamera.IsCameraBusy();
    }

    private void TakePicture(int maxSize)
    {
        NativeCamera.Permission permission = NativeCamera.TakePicture(( path) =>
        {
            DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeCam Start: " + path, false);
            if( path != null )
            {
                // Create a Texture2D from the captured image
                Texture2D texture = NativeCamera.LoadImageAtPath(path, maxSize, false, true);
                if(texture == null)
                {
                    DebugLog.getInstance().updateLog(LogType.NativeCamera, "Couldn't load texture: " + path, false);
                    return;
                }

                // Asign texture to the RawImage in the Scene
                ImageScript.showTexture();
                ImageScript.RotateImage(false, 0);
                ImageScript.changeCameraState(CameraState.Display);
                rawimage.texture = texture;

                // If a procedural texture is not destroyed manually, 
                // it will only be freed after a scene change
                // Save to gallery before Destroy
                StartCoroutine(SaveImage(texture));
                //Destroy(texture, 5f );
            }
            else
            {
                DebugLog.getInstance().updateLog(LogType.NativeCamera, "Cancel Image Picker");
                PanelDown.ToggleEditMode.Invoke(true, true);
            }
        }, maxSize );

        DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeCam End with: " + permission);
    }

    private void RecordVideo()
    {
        NativeCamera.Permission permission = NativeCamera.RecordVideo( ( path ) =>
        {
            DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeVideo Start: file://" + path, false);
            if( path != null )
            {
                // Play the recorded video
                Handheld.PlayFullScreenMovie("file://" + path);
            }
        } );

        DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeVideo End with: " + permission);
    }

    private IEnumerator SaveImage(Texture2D texture)
    {
        Texture2D textureInstance = new Texture2D(rawimage.texture.width, rawimage.texture.height, TextureFormat.ARGB32, false);
        textureInstance = texture;
        byte[] bytes = textureInstance.EncodeToPNG();

        yield return new WaitForEndOfFrame();

        NativeGallery.Permission permission = NativeGallery.SaveImageToGallery(
                                                                    textureInstance, 
                                                                    "CameraTest", 
                                                                    "CaptureImage.png", 
                                                                    (success, path) => 
                                                                    {
                                                                        Debug.Log(path);
                                                                        Debug.Log(path == "");
                                                                        DebugLog.getInstance().updateLog(
                                                                                                LogType.NativeGallery,
                                                                                                "Save Image To Gallery: " + 
                                                                                                success + " at " +
                                                                                                path 
                                                                        );
                                                                        RestFullAPI.TestAPI(bytes);
                                                                    }
        );
        Destroy(texture, 5f);  
    }

    private void PickImage( int maxSize )
    {
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery( ( path ) =>
        {
            Debug.Log( "Image path: " + path );
            if( path != null )
            {
                // Create Texture from selected image
                Texture2D texture = NativeGallery.LoadImageAtPath(path, maxSize, false, true);
                if( texture == null )
                {
                    Debug.Log( "Couldn't load texture from " + path );
                    return;
                }

                // Asign texture to the RawImage in the Scene
                ImageScript.showTexture();
                ImageScript.RotateImage(false, 0);
                ImageScript.changeCameraState(CameraState.Display);
                rawimage.texture = texture;
                byte[] bytes = texture.EncodeToPNG();
                
                //WSConnection.getInstance().setTextureData(texture.EncodeToPNG());
                RestFullAPI.TestAPI(bytes);
                Destroy(texture, 5f);
            }
            else
            {
                DebugLog.getInstance().updateLog(LogType.NativeCamera, "Cancel Image Picker");
                PanelDown.ToggleEditMode.Invoke(true, true);
            }
        } );

        Debug.Log( "Permission result: " + permission );
    }

    private void PickVideo()
    {
        NativeGallery.Permission permission = NativeGallery.GetVideoFromGallery( ( path ) =>
        {
            Debug.Log( "Video path: " + path );
            if( path != null )
            {
                // Play the selected video
                Handheld.PlayFullScreenMovie( "file://" + path );
            }
        }, "Select a video" );

        Debug.Log( "Permission result: " + permission );
    }
}
