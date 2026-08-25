import client from "./client";

export const predictChurn = async (customer) => {
  const { data } = await client.post("/predict", customer);
  return data;
};

export const predictChurnBatch = async (customers) => {
  const { data } = await client.post("/predict/batch", customers);
  return data;
};

export const getModelInfo = async () => {
  const { data } = await client.get("/model/info");
  return data;
};

export const getHealth = async () => {
  const { data } = await client.get("/health");
  return data;
};
