using UnityEngine;
using Newtonsoft.Json;
using System.Collections.Generic;

public class SMPLXAligner : MonoBehaviour
{
    public int MODEL_IMAGE_SIZE = 896;
    public string jsonFilePath;
    public GameObject smplxPrefab;
    public Camera alignmentCamera;
    public Texture2D backgroundImage;
    public float jointSize = 5f; // Tamaño de los círculos en píxeles
    private Vector2[] jointsScreenPositions; // Array para almacenar las posiciones de los joints en la pantalla
    
    private List<JSONReader.Data> parametersList;
    private string[] _smplxJointNames = new string[] { "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3", "jaw"};

    void Start()
    {
        JSONReader jsonReader = new JSONReader();
        jsonReader.ReadJSON(jsonFilePath);
        parametersList = jsonReader.GetData();

        if (parametersList != null)
        {
            foreach (var parameters in parametersList)
            {
                SetupCamera(parameters);
                AlignSMPLX(parameters);
                SetupBackgroundImage();
            }
        }
    }

    void SetupCamera(JSONReader.Data parameters)
    {
        if (parameters.camera.focalLength.Count >= 2 && parameters.camera.principalPoint.Count >= 2 && parameters.camera.imageSize.Count >= 2)
        {
            float fx = parameters.camera.focalLength[0];
            float fy = parameters.camera.focalLength[1];
            float cx = parameters.camera.principalPoint[0];
            float cy = parameters.camera.principalPoint[1];

            float fovY = 2 * Mathf.Atan(parameters.camera.imageSize[1] / (2 * fy)) * Mathf.Rad2Deg;
            alignmentCamera.fieldOfView = fovY;

            float aspect = (float)parameters.camera.imageSize[0] / parameters.camera.imageSize[1];
            alignmentCamera.aspect = aspect;
        }
        else
        {
            Debug.LogError("Camera parameters are not valid.");
        }
    }

    void AlignSMPLX(JSONReader.Data parameters)
    {
        if (parameters.location == null || parameters.location.Count < 2)
        {
            Debug.LogError("Location parameters are not valid.");
            return;
        }

        GameObject smplxInstance = Instantiate(smplxPrefab);
        SMPLX smplxComponent = smplxInstance.GetComponent<SMPLX>();

        Vector3 position = new Vector3(parameters.location[0], parameters.location[1], parameters.location[2]);
        smplxInstance.transform.position = position;

        if (parameters.rotation == null || parameters.rotation.Count < 3)
        {
            Debug.LogError("Rotation parameters are not valid.");
            return;
        }

        Vector3 rotationVector = new Vector3(parameters.rotation[0], parameters.rotation[1], parameters.rotation[2]);
        smplxInstance.transform.rotation = Quaternion.Euler(rotationVector * Mathf.Rad2Deg) * Quaternion.Euler(0, 180, 0);

        if (parameters.pose != null && parameters.pose.Count == _smplxJointNames.Length)
        {
            ApplyPose(smplxComponent, parameters.pose);
        }
        else
        {
            Debug.LogError("Pose parameters are not valid.");
        }

        if (parameters.shape != null && parameters.shape.Count == SMPLX.NUM_BETAS)
        {
            ApplyShape(smplxComponent, parameters.shape);
        }
        else
        {
            Debug.LogError("Shape parameters are not valid.");
        }

        if (parameters.expression != null && parameters.expression.Count == SMPLX.NUM_EXPRESSIONS)
        {
            ApplyExpression(smplxComponent, parameters.expression);
        }
        else
        {
            Debug.LogError("Expression parameters are not valid.");
        }

        if (parameters.joints2D != null)
        {
            CalculateJointsScreenPositions(parameters.joints2D, parameters);
            AlignWithJoint2D(smplxInstance, parameters.joints2D, (float)MODEL_IMAGE_SIZE, parameters.camera.imageSize[0], parameters.camera.imageSize[1], parameters.location[2]);
        }
    }

    private void ApplyPose(SMPLX smplxMannequin, List<List<float>> newPose) 
    {
        if (newPose.Count != _smplxJointNames.Length)
        {
            Debug.LogError($"Incorrect number of pose values: {newPose.Count}, expected: {_smplxJointNames.Length}");
            return;
        }

        for (int i = 0; i < newPose.Count; i++)
        {
            string jointName = _smplxJointNames[i];
            float rodX = newPose[i][0];
            float rodY = newPose[i][1];
            float rodZ = newPose[i][2];

            Quaternion rotation = SMPLX.QuatFromRodrigues(rodX, rodY, rodZ);
            smplxMannequin.SetLocalJointRotation(jointName, rotation);
        }

        smplxMannequin.EnablePoseCorrectives(true);
        Debug.Log("Pose applied successfully.");
    }

    void ApplyShape(SMPLX smplxMannequin, List<float> newBetas)
    {
        if (newBetas.Count != SMPLX.NUM_BETAS)
        {
            Debug.LogError($"Incorrect number of betas: {newBetas.Count}, expected: {SMPLX.NUM_BETAS}");
            return;
        }
        
        for (int i = 0; i < SMPLX.NUM_BETAS; i++)
        {
            smplxMannequin.betas[i] = newBetas[i];
        }
        smplxMannequin.SetBetaShapes();
        Debug.Log("Shape parameters applied successfully.");
    }

    void ApplyExpression(SMPLX smplxMannequin, List<float> newExpression) 
    {
        if (newExpression.Count != SMPLX.NUM_EXPRESSIONS)
        {
            Debug.LogError($"Incorrect number of expressions: {newExpression.Count}, expected: {SMPLX.NUM_EXPRESSIONS}");
            return;
        }
        
        for (int i = 0; i < SMPLX.NUM_EXPRESSIONS; i++)
        {
            smplxMannequin.expressions[i] = newExpression[i];
        }
        smplxMannequin.SetExpressions();
        Debug.Log("Expression parameters applied successfully.");
    }

    void AdjustPelvisPosition(GameObject smplxInstance, List<float> pelvisTranslation)
    {
        if (pelvisTranslation.Count >= 3)
        {
            Vector3 pelvisOffset = new Vector3(pelvisTranslation[0], pelvisTranslation[1], pelvisTranslation[2]);
            smplxInstance.transform.position += pelvisOffset;
        }
        else
        {
            Debug.LogError("Pelvis translation parameters are not valid.");
        }
    }

    void AlignWithJoint2D(GameObject smplxInstance, List<List<float>> joints_2d, float model_img_size, float image_width, float image_height, float translationZ)
    {
        Transform pelvisTransform = smplxInstance.GetComponent<SMPLX>().JointTransforms[_smplxJointNames[0]];
        Vector3 relativePositionPelvis = smplxInstance.transform.position - pelvisTransform.position;
        
        if (joints_2d.Count > 0 && joints_2d[0].Count >= 2)
        {
            Vector2 pelvis2D = new Vector2(joints_2d[0][0], joints_2d[0][1]);
            Vector2 pelvisViewport = new Vector2((pelvis2D.x - (model_img_size - image_width) / 2) / image_width, 1 - (pelvis2D.y / image_height));
            Ray ray = alignmentCamera.ViewportPointToRay(new Vector3(pelvisViewport.x, pelvisViewport.y, 0));
            float pelvisDistance = translationZ;
            Vector3 pelvis3DWorld = ray.GetPoint(pelvisDistance);
            Vector3 offset = pelvis3DWorld - smplxInstance.transform.position;
            smplxInstance.transform.position += offset + relativePositionPelvis;

            Debug.Log($"Pelvis 2D: {pelvis2D}, Viewport: {pelvisViewport}, 3D World: {pelvis3DWorld}");
        }
        else
        {
            Debug.LogError("Joint 2D parameters are not valid.");
        }
    }

    void SetupBackgroundImage()
    {
        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.transform.position = new Vector3(0, 0, 10);
        
        float aspectRatio = (float)backgroundImage.width / backgroundImage.height;
        float quadHeight = 2f * Mathf.Tan(alignmentCamera.fieldOfView * 0.5f * Mathf.Deg2Rad) * quad.transform.position.z;
        float quadWidth = quadHeight * aspectRatio;
        
        quad.transform.localScale = new Vector3(quadWidth, quadHeight, 1);
        
        Material mat = new Material(Shader.Find("Unlit/Texture"));
        mat.mainTexture = backgroundImage;
        quad.GetComponent<Renderer>().material = mat;
    }

    void CalculateJointsScreenPositions(List<List<float>> joints_2d, JSONReader.Data parameters)
    {
        if (joints_2d == null || parameters.camera.imageSize.Count < 2)
        {
            Debug.LogError("Joint 2D parameters or camera image size are not valid.");
            return;
        }

        jointsScreenPositions = new Vector2[joints_2d.Count];

        float scaleX = (float)Screen.width / parameters.camera.imageSize[0];
        float scaleY = (float)Screen.height / parameters.camera.imageSize[1];

        for (int i = 0; i < joints_2d.Count; i++)
        {
            if (joints_2d[i].Count >= 2)
            {
                float x = (joints_2d[i][0] - (MODEL_IMAGE_SIZE - parameters.camera.imageSize[0]) / 2) * scaleX;
                float y = (parameters.camera.imageSize[1] - joints_2d[i][1]) * scaleY;
                jointsScreenPositions[i] = new Vector2(x, y);

                if (i == 0) Debug.Log($"Pelvis jointsScreenPositions[{i}]: {jointsScreenPositions[i]}");
            }
            else
            {
                Debug.LogError($"Joint 2D data at index {i} is not valid.");
            }
        }
    }

    void OnGUI()
    {
        if (jointsScreenPositions == null) return;

        GUIStyle circleStyle = new GUIStyle();
        Texture2D texture = new Texture2D(1, 1);
        texture.SetPixel(0, 0, Color.green);
        texture.Apply();
        circleStyle.normal.background = texture;

        for (int i = 0; i < jointsScreenPositions.Length; i++)
        {
            Vector2 pos = jointsScreenPositions[i];
            pos.y = Screen.height - pos.y;
            Rect rect = new Rect(pos.x - jointSize / 2, pos.y - jointSize / 2, jointSize, jointSize);
            GUI.Box(rect, GUIContent.none, circleStyle);
        }
    }
}
