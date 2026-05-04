defmodule LangChain.Utils.VertexAnthropicConfig do
  @moduledoc """
  Configuration for accessing Anthropic Claude models via Google Cloud Vertex
  AI's Model Garden.

  Anthropic Claude on Vertex uses a Google-authenticated endpoint at
  `https://{region}-aiplatform.googleapis.com/...` instead of Anthropic's
  native `api.anthropic.com`. The body is the standard Anthropic Messages
  API body except: the `model` is part of the URL (not the body) and the
  body must include `anthropic_version` set to a Vertex-recognised value
  (default `"vertex-2023-10-16"`).

  Authentication is a gcloud bearer token rather than an `x-api-key`
  header — supply a zero-argument function in `:credentials` that returns
  a fresh access token each call (so callers can refresh tokens as
  needed).

  ## Example

      ChatAnthropic.new!(%{
        model: "claude-haiku-4-5@20251001",
        vertex: %{
          credentials: fn -> {:ok, %{token: token}} = MyApp.gcloud_token(); token end,
          project_id: "my-gcp-project",
          region: "us-east5"
        }
      })

  Or load from application config:

      config :langchain,
        gcp_project_id: System.fetch_env!("GCP_PROJECT_ID"),
        gcp_region: System.get_env("GCP_REGION", "us-east5"),
        gcp_token_provider: {MyApp, :gcloud_token, []}

      ChatAnthropic.new!(%{
        model: "claude-haiku-4-5@20251001",
        vertex: VertexAnthropicConfig.from_application_env!()
      })
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    # Zero-argument function returning a Google Cloud access token (a string).
    # Called on every request so the caller can refresh as needed.
    field :credentials, :any, virtual: true
    field :project_id, :string
    field :region, :string, default: "us-east5"
    field :anthropic_version, :string, default: "vertex-2023-10-16"
  end

  @type t :: %VertexAnthropicConfig{}

  @create_fields [:credentials, :project_id, :region, :anthropic_version]
  @required_fields [:credentials, :project_id]

  def changeset(config, attrs) do
    config
    |> cast(attrs, @create_fields)
    |> validate_required(@required_fields)
  end

  @doc """
  Returns the bearer token by invoking the configured credentials function.
  """
  @spec token(t()) :: String.t()
  def token(%VertexAnthropicConfig{credentials: credentials}) when is_function(credentials, 0) do
    credentials.()
  end

  @doc """
  Builds the Vertex AI predict URL for the given model and stream mode.

  Per Vertex AI docs, the model name passes through the URL with `@version`
  suffix preserved (e.g. `claude-haiku-4-5@20251001`). The action is
  `:streamRawPredict` for streamed responses, `:rawPredict` otherwise.
  """
  @spec url(t(), keyword()) :: String.t()
  def url(%VertexAnthropicConfig{region: region, project_id: project_id}, opts) do
    model = Keyword.fetch!(opts, :model)
    stream = Keyword.get(opts, :stream, false)
    action = if stream, do: "streamRawPredict", else: "rawPredict"

    "https://#{region}-aiplatform.googleapis.com/v1/projects/#{project_id}" <>
      "/locations/#{region}/publishers/anthropic/models/#{model}:#{action}"
  end

  @doc """
  Loads the Vertex Anthropic config from `:langchain` application env.

  Expects:
    * `:gcp_project_id` — string
    * `:gcp_region` — string (defaults to "us-east5")
    * `:gcp_token_provider` — `{module, function, args}` MFA returning a
      bearer token string
  """
  @spec from_application_env!() :: t()
  def from_application_env! do
    {mod, fun, args} =
      Application.fetch_env!(:langchain, :gcp_token_provider)

    %VertexAnthropicConfig{
      credentials: fn -> apply(mod, fun, args) end,
      project_id: Application.fetch_env!(:langchain, :gcp_project_id),
      region: Application.get_env(:langchain, :gcp_region, "us-east5")
    }
  end
end
