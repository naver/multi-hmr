using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

public class SMPLXAligner : MonoBehaviour
{
    public Camera mainCamera;
    public GameObject smplxPrefab;
    public Texture2D backgroundImage;
    public string jsonFilePath;

    public Vector3 manualOffset = new Vector3(0.077f, -1.533f, 0.31f);
    private GameObject instanciaSmplx;

    private List<GameObject> smplxInstances = new List<GameObject>();
    private string[] _smplxJointNames = new string[] { "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3", "jaw" };
    private Vector3 initialPosition;
    private bool canBeUpdated = false;

    void Start()
    {
        canBeUpdated = false;

        List<SMPLXParams> smplxParamsList = JSONReader.ReadJSONFile(jsonFilePath);

        if (smplxParamsList != null && smplxParamsList.Count > 0 && smplxParamsList[0].camera != null)
        {
            SetupCamera(smplxParamsList[0].camera);
        }
        else
        {
            Debug.LogWarning("No se encontraron parámetros de cámara en el archivo JSON.");
            SetupDefaultCamera();
        }

        SetupBackgroundImage();
        CreateAndAlignSMPLX(smplxParamsList);
    }

    void Update()
    {
        if (canBeUpdated) instanciaSmplx.transform.position = initialPosition + manualOffset;
    }

    void SetupCamera(CameraParams cameraParams)
    {
        if (cameraParams == null || cameraParams.imageSize == null || cameraParams.focalLength == null)
        {
            Debug.LogError("Camera parameters are not properly defined.");
            return;
        }

        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 2 * Mathf.Atan(cameraParams.imageSize[1] / (2 * cameraParams.focalLength[1])) * Mathf.Rad2Deg;
        mainCamera.aspect = (float)cameraParams.imageSize[0] / cameraParams.imageSize[1];
        mainCamera.nearClipPlane = 0.1f;
        mainCamera.farClipPlane = 1000f;
    }

    void SetupDefaultCamera()
    {
        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 59f;
        mainCamera.nearClipPlane = 0.1f;
        mainCamera.farClipPlane = 1000f;
    }

    void SetupBackgroundImage()
    {
        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.transform.position = new Vector3(0, 0, 10);

        float aspectRatio = (float)backgroundImage.width / backgroundImage.height;
        float quadHeight = 2f * Mathf.Tan(mainCamera.fieldOfView * 0.5f * Mathf.Deg2Rad) * quad.transform.position.z;
        float quadWidth = quadHeight * aspectRatio;

        quad.transform.localScale = new Vector3(quadWidth, quadHeight, 1);

        Material mat = new Material(Shader.Find("Unlit/Texture"));
        mat.mainTexture = backgroundImage;
        quad.GetComponent<Renderer>().material = mat;
    }

    void CreateAndAlignSMPLX(List<SMPLXParams> smplxParamsList)
    {
        if (smplxParamsList == null)
        {
            Debug.LogError("smplxParamsList es nulo.");
            return;
        }

        foreach (var parames in smplxParamsList)
        {
            GameObject smplxInstance = Instantiate(smplxPrefab);

            if (smplxInstance == null)
            {
                Debug.LogError("La instancia de smplxPrefab es nula.");
                continue;
            }

            smplxPrefab.SetActive(false);
            instanciaSmplx = smplxInstance;
            smplxInstances.Add(smplxInstance);

            ApplySMPLXParams(smplxInstance, parames);
            if (parames.locationV != null)
            {
                AlignWithCamera(smplxInstance, parames);
            }
            else
            {
                Debug.LogWarning("parames.locationV es nulo. Se omite el alineamiento para esta instancia.");
            }

            initialPosition = instanciaSmplx.transform.position;
            canBeUpdated = true;
        }
    }

    void ApplySMPLXParams(GameObject smplxInstance, SMPLXParams parames)
    {
        SMPLX smplxComponent = smplxInstance.GetComponent<SMPLX>();
        if (smplxComponent != null && parames.humans != null && parames.humans.Length > 0 && parames.humans[0] != null)
        {
            if (parames.humans[0].rotation_vector != null)
                SetPose(smplxComponent, parames.humans[0].rotation_vector);

            if (parames.humans[0].expression != null)
                SetExpression(smplxComponent, parames.humans[0].expression);

            if (parames.humans[0].shape != null)
                SetShape(smplxComponent, parames.humans[0].shape);
        }
        else
        {
            Debug.LogWarning("SMPLXParams.humans is null or empty, or the first element is null.");
        }
    }

    void AlignWithCamera(GameObject smplxInstance, SMPLXParams parames)
    {
        if (smplxInstance == null)
        {
            Debug.LogError("smplxInstance es nulo.");
            return;
        }

        if (parames == null)
        {
            Debug.LogError("parames es nulo.");
            return;
        }

        if (parames.locationV == null)
        {
            Debug.LogError("parames.locationV es nulo.");
            return;
        }

        Vector3 worldPosition;
        if (parames.camera != null)
        {
            if (parames.camera.imageSize == null)
            {
                Debug.LogError("parames.camera.imageSize es nulo.");
                return;
            }

            Vector2 imageSize = new Vector2(parames.camera.imageSize[0], parames.camera.imageSize[1]);
            worldPosition = CalculateWorldPosition(parames.locationV, imageSize, parames.distance);
        }
        else
        {
            Vector2 imageSize = new Vector2(backgroundImage.width, backgroundImage.height);
            worldPosition = CalculateWorldPosition(parames.locationV, imageSize, parames.distance);
        }

        ApplyTransformations(smplxInstance, worldPosition, parames);
    }

    Vector3 CalculateWorldPosition(float[] location, Vector2 imageSize, float distance)
    {
        if (location == null || imageSize == null)
        {
            Debug.LogError("location o imageSize es nulo.");
            return Vector3.zero;
        }

        Vector2 normalizedPosition = new Vector2(location[0] / imageSize.x, location[1] / imageSize.y);
        return mainCamera.ViewportToWorldPoint(new Vector3(normalizedPosition.x, 1 - normalizedPosition.y, distance));
    }

    void ApplyTransformations(GameObject smplxInstance, Vector3 worldPosition, SMPLXParams parames)
    {
        if (parames.rotationV == null || parames.positionOffsetV == null || parames.rotationOffsetV == null)
        {
            Debug.LogError("Uno o más parámetros de transformación son nulos.");
            return;
        }

        Quaternion worldRotation = Quaternion.Euler(parames.rotationV[0], -parames.rotationV[1], -parames.rotationV[2] + 180f);

        smplxInstance.transform.rotation = worldRotation;
        smplxInstance.transform.position = worldPosition;

        smplxInstance.transform.position += smplxInstance.transform.TransformDirection(new Vector3(parames.positionOffsetV[0], parames.positionOffsetV[1], parames.positionOffsetV[2]));
        smplxInstance.transform.rotation *= Quaternion.Euler(new Vector3(parames.rotationOffsetV[0], parames.rotationOffsetV[1], parames.rotationOffsetV[2]));

        initialPosition = smplxInstance.transform.position;
        canBeUpdated = true;
    }

    private void SetShape(SMPLX smplxMannequin, float[] newBetas)
    {
        if (newBetas == null || newBetas.Length != SMPLX.NUM_BETAS)
        {
            Debug.LogError("No tiene el número correcto de betas o es nulo: " + (newBetas == null ? "null" : newBetas.Length.ToString()) + " porque deberían ser: " + SMPLX.NUM_BETAS);
            return;
        }

        for (int i = 0; i < SMPLX.NUM_BETAS; i++)
        {
            smplxMannequin.betas[i] = newBetas[i];
        }
        smplxMannequin.SetBetaShapes();
        Debug.Log("Se han aplicado correctamente los blendshapes.");
    }

    private void SetPose(SMPLX smplxMannequin, float[][] newPose)
    {
        if (newPose == null || newPose.Length != _smplxJointNames.Length)
        {
            Debug.LogError("No tiene el número correcto de valores o es nulo: " + (newPose == null ? "null" : newPose.Length.ToString()) + " porque deberían ser: " + _smplxJointNames.Length);
            return;
        }

        for (int i = 0; i < newPose.Length; i++)
        {
            string jointName = _smplxJointNames[i];
            float rodX = newPose[i][0];
            float rodY = newPose[i][1];
            float rodZ = newPose[i][2];

            Quaternion rotation = SMPLX.QuatFromRodrigues(rodX, rodY, rodZ);
            smplxMannequin.SetLocalJointRotation(jointName, rotation);
        }

        smplxMannequin.EnablePoseCorrectives(true);
        Debug.Log("Se ha aplicado correctamente la pose.");
    }

    private void SetExpression(SMPLX smplxMannequin, float[] newExpression)
    {
        if (newExpression == null || newExpression.Length != SMPLX.NUM_EXPRESSIONS)
        {
            Debug.LogError("No tiene el número correcto de expresiones o es nulo: " + (newExpression == null ? "null" : newExpression.Length.ToString()) + " porque deberían ser: " + SMPLX.NUM_EXPRESSIONS);
            return;
        }

        for (int i = 0; i < SMPLX.NUM_EXPRESSIONS; i++)
        {
            smplxMannequin.expressions[i] = newExpression[i];
        }
        smplxMannequin.SetExpressions();

        Debug.Log("Se han aplicado correctamente las expresiones.");
    }
}
