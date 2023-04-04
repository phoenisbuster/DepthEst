using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;
using TMPro;

//"http://localhost:5000/upload"
public class RestFullAPI : MonoBehaviour
{   
    public TextMeshProUGUI ResultDisplay;

    public static RestFullAPI _instance;

    private void Awake() 
    {
        _instance = this;
    }

    public static async void TestAPI(byte[] content) 
    {
        await Main(content);
    }
    
    static async Task Main(byte[] content)
    {
        try
        {    
            _instance.setResultDisplay("Loading...");
            var httpClient = new HttpClient();
            var requestContent = new MultipartFormDataContent();
            var fileContent = new ByteArrayContent(content);
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
            requestContent.Add(fileContent, "image", "image.png");

            httpClient.DefaultRequestHeaders.Add("X-CSRF-Token", "my-secret-key");

            httpClient.Timeout = TimeSpan.FromSeconds(10);
            Debug.Log(httpClient.Timeout);

            var fullUrl = "http://" + WSConnection.getInstance().address + "/upload";
            Debug.Log(fullUrl);

            var response = await httpClient.PostAsync(fullUrl, requestContent);
            Debug.Log(response.StatusCode);
            var responseContent = await response.Content.ReadAsStringAsync();
            switch(response.StatusCode)
            {
                case System.Net.HttpStatusCode.OK:
                    Debug.Log(responseContent);
                    //JObject json = JObject.Parse(responseContent);
                    _instance.setResultDisplay(responseContent);
                    break;

                case System.Net.HttpStatusCode.Forbidden:
                case System.Net.HttpStatusCode.BadRequest:
                    throw new Exception(responseContent);
            }
            
        }
        catch(Exception e)
        {
            Debug.LogWarning("Unable to connect to Server: " + e.Message);
            _instance.setResultDisplay(e.Message);
        }
        finally
        {
            PanelDown.ToggleEditMode.Invoke(true, true);
        }
    }

    public void setResultDisplay(string value = "")
    {
        ResultDisplay.text = value;
    }
}
